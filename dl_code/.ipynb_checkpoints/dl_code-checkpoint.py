import torch
import typing


class BERT(torch.nn.Module):
    """
    The class for a BERT model to be applied to
    tokenized representations of patient feature
    data. This model serves as a context-aware embedding
    model and performs classification on its 'start of sequence'
    token, often referred to as a CLS token. CLS tokens
    are typically meant to represent the entire input.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        num_tokens: int,
        seq_len: int,
        embedding_dim: int,
        targets: typing.List = []
    ):
        """
        Initializing the BERT model class.

        :param encoder: torch.nn.Module. This is the MHA
            encoder.
        :param decoder: None. This is not allowed for a BERT
            model and is kept in to enable consistency across
            model types.
        :param num_tokens: Int, this is the size of the vocabulary.
        :param seq_len: Int, context length.
        :param embedding_dim: Int, The embedding dimension of the tokens.
        :param targets: List, identifying the regression or classification
            targets being created.
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.targets = targets

        self.embedder = torch.nn.Embedding(
            num_tokens, embedding_dim
        ).to(self.device)

        self.encoder = encoder

        if decoder is not None:
            raise ValueError("BERT model does not take a decoder.")

        self.unmasker = torch.nn.Linear(
            self.encoder.d_model, num_tokens
        ).to(self.device)

        self.predictors = {}
        for target in self.targets:
            self.predictors[target] = torch.nn.Linear(
                self.encoder.d_model, 1
            ).to(self.device)

    def forward(
        self,
        batch: typing.Dict
    ) -> typing.Dict:
        """
        Forward pass for the BERT model.

        :param batch: Dictionary, with the following required
            keys: "tokens", "masked_tokens", "times", "padding_mask".

        :return: output: Dictionary, with the following keys.
            "h": Final embedding of the input visit trajectory.
                h.shape = [batch_size, seq_len, d_model]
            "unmasked": Prediction on the values of the tokens
                of the model.
                unmasked.shape = [batch_size, seq_len, num_tokens]
            A key for each target in self.targets. Each one will have shape
                [batch_size, 1].
        """

        if (
            "tokens" not in batch.keys()
            or
            "masked_tokens" not in batch.keys()
            or
            "times" not in batch.keys()
            or
            "padding_mask" not in batch.keys()
        ):
            raise AssertionError("There is a missing key in the batch")

        if self.training:
            X = batch['masked_tokens'].to(self.device)
        else:
            X = batch['tokens'].to(self.device)

        ts = batch['times'].to(self.device)
        padding_mask = batch['padding_mask'].to(self.device)

        embedded = self.embedder(X)

        encoder_output = self.encoder(embedded, ts, padding_mask)

        h = encoder_output['h']

        unmasked = self.unmasker(h)

        regression_targets = {}

        for target in self.targets:
            regression_targets[target] = (
                self.predictors[target](h[:, 0, :])
            )

        output = {}

        for target in self.targets:
            output[target] = regression_targets[target]
        output['h'] = h
        output['unmasked'] = unmasked

        return output

import torch
import typing
import itertools
import patient_feature_learning.models.layers as pfl_layers


class Encoder(torch.nn.Module):
    """
    Implementing a base encoder module for an encoder or an encoder-decoder
    model. It encodes the mean and log variance of the encoding, to be
    used in downstream sampling efforts from a latent space.
    """
    def __init__(
        self,
        input_dimension: int,
        encoder_dimensions: typing.List[int],
        latent_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU()
    ):
        """
        Base encoder layer. As is, this implements a
        multi-layer perceptron, whose hidden dimensions
        are specified in encoder dimensions.

        :param encoder_dimensions: list, specifies the hidden dimensions
            of each layer of the encoder.
        :param input_dimensions: int, dimension of the input space.
        :param latent_dim: int, size of the latent space.
        :param activation: torch.nn.Module, specifies the activation.
            Defaults to ReLU.
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in itertools.pairwise(
            itertools.chain([input_dimension], encoder_dimensions)
        ):
            layers.extend(
                (torch.nn.Linear(in_dim, out_dim),
                 activation)
             )
        self.model = torch.nn.Sequential(*layers)
        self.z_mu = torch.nn.Linear(encoder_dimensions[-1], latent_dim)
        self.z_log_var = torch.nn.Linear(encoder_dimensions[-1], latent_dim)

    def forward(
        self,
        X: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method of the encoder.

        :param X: tensor, shape = [batch_size, .., input_dimension]
        :return: z_mu, z_log_var, the mean and log variance of the embedding.
            z_mu: z_mu.shape = [batch_size, ..., latent_dim]
            z_log_var: z_log_var.shape = z_mu.shape
        """
        h = self.model(X)
        return {
            'z_mu': self.z_mu(h),
            'z_log_var': self.z_log_var(h)
        }


class Decoder(torch.nn.Module):
    """
    Implementing a base decoder module for an encoder-decoder style of model.
    It translates from the latent dimension to the original input space.
    """
    def __init__(
        self,
        latent_dim: int,
        decoder_dimensions: typing.List[int],
        output_dimension: int,
        activation: torch.nn.Module = torch.nn.ReLU()
    ):
        """
        Base decoder layer. Initializes an MLP decoder layer.
        :param decoder_dimensions: list, specifies the hidden
            dimensions of each layer of the decoder.
        :param latent_dimension: int, size of the latent
            space.
        :param output_dimensions: int, size of the output
            space.
        :param activation: torch.nn.Module = torch.nn.ReLU()
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in itertools.pairwise(
            itertools.chain([latent_dim], decoder_dimensions)
        ):
            layers.extend(
                (torch.nn.Linear(in_dim, out_dim),
                 activation)
             )
        self.model = torch.nn.Sequential(*layers)
        self.output = torch.nn.Linear(decoder_dimensions[-1], output_dimension)

    def forward(
        self,
        z: torch.Tensor
    ):
        """
        Forward pass of the decoder.

        :param z: z.shape = [batch_size, ..., latent_dim]
        :return Xhat: Prediction of the input.
            Xhat.shape = [batch_size, ..., output_dimensions]
        """

        logits = self.output(self.model(z))
        Xhat = torch.nn.Sigmoid()(logits)
        return {
            'Xhat': Xhat
        }


class Encoder_VADER(torch.nn.Module):
    """
    Encoder for the vader method.
    Applies the imputer and iteratively
    applies the peephole LSTM.
    See variational_deep_embedding.VaDER class for reference.
    """
    def __init__(
        self,
        time_steps: int,
        input_size: int,
        hidden_layer: int,
        latent_dim: int,
        activation=torch.nn.Softplus()
    ):
        """
        Initializing the encoder for VADER. In particular,
        preparing the imputation layer, the activation,
        and the Peephole LSTM layer. This can be improved
        by allowing for the encoder to have other possible
        rnn types.

        :param time_steps: int, Number of time steps in each time series.
        :param input_size: int, Dimension of each time point in the series.
        :param hidden_layer: int, Dimension of the cell state and hidden state
            of the rnn.
        :param latent_dim: int, Dimension of the latent space.
        :param activation: torch.nn.Module, activation for the encoder.
        """
        super().__init__()
        self.time_steps = time_steps
        self.hidden_layer = hidden_layer
        self.activation = activation

        self.impute_layer = pfl_layers.ImputationLayer(input_size)
        self.lstm_layer = pfl_layers.PeepholeLSTMCell(
            input_size, hidden_size=hidden_layer
        )
        self.z_mu = torch.nn.Linear(hidden_layer, latent_dim)
        self.z_log_var = torch.nn.Linear(hidden_layer, latent_dim)

    def forward(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the encoder. Note that
        we iteratively run the LSTM layer over the time
        series, using only the final cell state as the input
        to the latent space.

        :param X: Input tensor, X.shape = [batch_size, time_steps, input_size].
        :param missing_mask: Binary tensor indicating what
        values were missing in X. missing_mask.shape = X.shape.

        :return: z_mu, z_log_var.
            z_mu.shape = [batch_size, latent_dim].
            z_mu is the mean embedding of the input.
            z_log_var.shape = z_mu.shape.
            z_log_var is the variance of each embedding.
        """
        imputed_X = self.impute_layer(X, missing_mask)
        h = torch.zeros(X.shape[0], self.hidden_layer)
        c = torch.zeros(X.shape[0], self.hidden_layer)
        for i in range(self.time_steps):
            h, c = self.lstm_layer(imputed_X[:, i, :], (h, c))
        c = self.activation(c)
        return {
            'z_mu': self.z_mu(h),
            'z_log_var': self.z_log_var(h)
        }


class Decoder_VADER(torch.nn.Module):
    """
    Decoder for Vader method.
    Applies peephole lstm method
    iteratively, storing the cell
    state along the way. The cell state
    is then decoded back into the original input space.
    See variational_deep_embedding.VaDER class for reference.
    """
    def __init__(
        self,
        time_steps: int,
        input_size: int,
        hidden_layer: int,
        latent_dim: int,
        activation=torch.nn.Softplus()
    ):
        """
        Initializing the decoder. Similar to Encoder_VADER

        :param time_steps: int, Number of time steps in each time series.
        :param input_size: int, Dimension of each time point in the series.
        :param hidden_layer: int, Dimension of the cell state and hidden state
            of the rnn.
        :param latent_dim: int, Dimension of the latent space.
        :param activation: torch.nn.Module, activation for the encoder.
        """
        super().__init__()
        self.time_steps = time_steps
        self.activation = activation
        self.hidden_layer = hidden_layer
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.latent_to_hidden = torch.nn.Linear(latent_dim, hidden_layer)
        self.lstm_layer = pfl_layers.PeepholeLSTMCell(input_size, hidden_layer)
        self.final_decode = torch.nn.Linear(hidden_layer, input_size)

    def forward(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs the forward pass of the decoder, accumulating a cell state
        per time point in the time series being reconstructed. The cell state
        is what's used for the reconstruction in the end.

        :param z: Latent embeddings, z.shape = [batch_size, latent_dim]

        :return: Reconstruction of time series,
            reconstruction.shape = [batch_size, time_steps, input_size].
        """
        hidden_state = self.latent_to_hidden(z)
        hidden_state = self.activation(hidden_state)

        cell_state = torch.zeros(
            hidden_state.size(), dtype=z.dtype, device=z.device
        )
        inputs = torch.zeros(
            (
                z.shape[0], self.time_steps, self.input_size
            ), dtype=z.dtype, device=z.device
        )

        hidden_state_collector = torch.empty(
            (
                z.shape[0], self.time_steps, self.hidden_layer
            ),  dtype=z.dtype, device=z.device
        )

        for i in range(self.time_steps):
            x = inputs[:, i, :]
            hidden_state, cell_state = self.lstm_layer(
                x, (hidden_state, cell_state)
            )
            hidden_state_collector[:, i, :] = hidden_state

        Xhat = self.final_decode(hidden_state_collector)

        return {
            'Xhat': Xhat
        }


class Transformer_Encoder(torch.nn.Module):
    """
    Encoder model using MHA and feedforward layers
    to parse through the data.
    """
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_model: int,
        layers: int,
        heads: int,
        latent_dim: int,
        seq_len: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        dropout: float = 0.0
    ):
        """
        Initialize the MHA Encoder Class.
        Relies on :class:layers.MultiHeadedAttention.

        :param d_input: Int, indicates the dimensionality of the input.
        :param d_hidden: Int, indicates the dimensionality of the hidden layers
            within the attention module.
        :param d_model: Int, indicates the dimensionality of the model,
            connecting layer to layer.
        :param layers: Int, Number of attention and feedforward layers to
            construct.
        :param heads: Int, number of attention heads to include.
        :param latent_dim: Int, the size of the latent space - final output of
            the encoder.
        :param seq_len: Int, the max length of the sequence being feed
            through.
        :param activation: Torch.nn.Module. The activation layer size.
        :param dropout: float. Parameter that controls the amount of dropout
            in each layer.
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.layers = layers
        self.heads = heads
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.feedforward = torch.nn.Linear(
            self.d_input, self.d_model
        ).to(self.device)

        self.activation = activation.to(self.device)

        self.norm = pfl_layers.AddNorm(self.d_model).to(self.device)

        self.dropout = torch.nn.Dropout(dropout)

        self.position_embedder = pfl_layers.PositionalEmbeddingMatrix(
            self.d_model
        ).to(self.device)

        self.MHA = []
        self.MLP = []

        for layer in range(self.layers):
            mha_layer = pfl_layers.MultiHeadAttention(
                self.heads,
                self.d_model,
                self.d_model,
                self.d_model,
                self.d_hidden,
                self.d_model
            ).to(self.device)
            self.MHA.append(mha_layer)

        for layer in range(self.layers):
            ff_layer = [
                torch.nn.Linear(d_model, d_model),
                self.activation,
                torch.nn.Linear(d_model, d_model)
            ]
            ff_layer = torch.nn.Sequential(*ff_layer).to(self.device)
            self.MLP.append(ff_layer)

        self.z_mu = torch.nn.Linear(
            self.d_model * self.seq_len, self.latent_dim
        ).to(self.device)
        self.z_log_var = torch.nn.Linear(
            self.d_model * self.seq_len, self.latent_dim
        ).to(self.device)

        self.MHA = torch.nn.ModuleList(self.MHA)
        self.MLP = torch.nn.ModuleList(self.MLP)

    def forward(
        self,
        x: torch.Tensor,
        ts: typing.Optional[torch.Tensor] = None,
        padding_mask: typing.Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Running the forward pass of the multihead attention encoder.

        :param x: Input tensor, representing patient visits.
            x.shape = [batch_size, seq_len, d_input]
        :param ts: Optional, Time index tensor, representing the
            time at which a visit occurs. If none, populate with
            torch.arange() for seq len.
            ts.shape = [batch_size, seq_len]
        :param padding_mask: Input tensor, representing what visits
            are padding. Is optional.
            padding_mask.shape = ts.shape

        :return:
            mha_encoded: Encoded multi-headed attention state.
                mha_encoded.shape = [batch_size, seq_len, d_model].
            z_mu: Mean embedding for a lower dimensional latent space
                that aggregates data over sequence length.
            z_log_var: Log variance of embedding.
                z_mu.shape = z_log_var.shape = [batch_size, d_latent].
        """

        h = self.feedforward(x)
        if ts is None:
            ts = torch.arange(
                self.seq_len
            ).unsqueeze(
                0
            ).repeat(
                x.shape[0], 1
            ).to(self.device)

        h = h + self.position_embedder(ts)

        for layer in range(self.layers):
            next_h, _ = self.MHA[layer](h, h, h, padding_mask)
            next_h = self.dropout(next_h)
            next_h = self.activation(next_h)
            h = self.norm(next_h, h)
            next_h = self.MLP[layer](h)
            next_h = self.dropout(next_h)
            h = self.norm(next_h, h)

        # Shape from [batch_size, seq_len, d_model] to
        # [batch_size, seq_len * d_model]
        unfolded_h = h.view(h.shape[0], h.shape[1]*h.shape[2])

        return {
            'h': h,
            'z_mu': self.z_mu(unfolded_h),
            'z_log_var': self.z_log_var(unfolded_h)
        }


class Transformer_Decoder(torch.nn.Module):
    """
    Initializing the MHA decoder.
    """
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_model: int,
        layers: int,
        heads: int,
        latent_dim: int,
        seq_len: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        tokenized: bool = False,
        dropout: float = 0.0
    ):
        """
        Initializing the class.

        :param d_input: Int, Dimension of the input space.
        :param d_hidden: Int, Dimension of the hidden space.
        :param d_model: Int, Dimension of the model.
        :param layers: Int, number of model layers.
        :param heads: Int, number of heads.
        :param latent_dim: Int, size of the latent space.
        :param seq_len: Int, length of the sequence.
        :param activation: torch.nn.Module, activation layer.
        :param tokenized: bool, indicates whether input is in tokenized state.
        :param dropout: float, Parameter that controls the amount of dropout
            in each layer.
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.layers = layers
        self.heads = heads
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.activation = activation
        self.tokenized = tokenized

        self.feedforward = torch.nn.Linear(
            self.d_input,
            self.d_model
        ).to(self.device)

        self.dropout = torch.nn.Dropout(dropout)

        self.latent_to_model = torch.nn.Linear(
            self.latent_dim,
            self.seq_len * self.d_model
        ).to(self.device)

        self.norm = pfl_layers.AddNorm(self.d_model).to(self.device)

        self.model_to_input = torch.nn.Linear(
            self.seq_len * d_model,
            self.seq_len * d_input
        ).to(self.device)

        self.position_embedder = pfl_layers.PositionalEmbeddingMatrix(
            self.d_model
        ).to(self.device)

        self.self_MHA = []
        self.cross_MHA = []
        self.MLP = []

        for layer in range(self.layers):
            mha_layer = pfl_layers.MultiHeadAttention(
                self.heads,
                self.d_model,
                self.d_model,
                self.d_model,
                self.d_hidden,
                self.d_model
            ).to(self.device)
            self.self_MHA.append(mha_layer)

            mha_layer = pfl_layers.MultiHeadAttention(
                self.heads,
                self.d_model,
                self.d_model,
                self.d_model,
                self.d_hidden,
                self.d_model
            ).to(self.device)
            self.cross_MHA.append(mha_layer)

        for layer in range(self.layers):
            ff_layer = [
                torch.nn.Linear(d_model, d_model),
                self.activation,
                torch.nn.Linear(d_model, d_model)
            ]
            ff_layer = torch.nn.Sequential(*ff_layer).to(self.device)
            self.MLP.append(ff_layer)

        self.self_MHA = torch.nn.ModuleList(self.self_MHA)
        self.cross_MHA = torch.nn.ModuleList(self.cross_MHA)
        self.MLP = torch.nn.ModuleList(self.MLP)

    def forward(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the decoder. Decodes all
        visits simultaneously, given a latent state.

        TODO: Thinking about how to do autoregressive generation
        with tokens in the same module. Should I jerry rig it here?
        or should I do something else? Currently, only works for
        visit level representations.

        :param z: Latent embedding used for cross attention.
            z.shape = [batch_size, latent_dim]

        :return:
            init_X: Probability distribution ove the codes for
            each visit. In the visit-level representation,
            init_X.shape = [batch_size, seq_len, d_input]
        """

        projected_z = self.latent_to_model(z)
        projected_z = projected_z.reshape(
            projected_z.shape[0], self.seq_len, self.d_model)

        ts = torch.arange(
            self.seq_len
        ).unsqueeze(
            0
        ).repeat(
            z.shape[0], 1
        ).to(self.device)

        h = projected_z + self.position_embedder(ts)

        for layer in range(self.layers):
            next_h, _ = self.self_MHA[layer](h, h, h)
            next_h = self.dropout(next_h)
            next_h = self.activation(next_h)
            h = self.norm(h, next_h)

            next_h, _ = self.cross_MHA[layer](
                h,
                projected_z,
                projected_z
            )
            next_h = self.dropout(next_h)
            next_h = self.activation(next_h)
            h = self.norm(h, next_h)
            next_h = self.MLP[layer](h)
            next_h = self.dropout(next_h)
            h = self.norm(h, next_h)

        h = h.reshape(h.shape[0], self.seq_len * self.d_model)
        init_X = self.model_to_input(h)
        init_X = init_X.reshape(init_X.shape[0], self.seq_len, self.d_input)

        return {
            'Xhat': init_X
        }

import torch
import typing

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImputationLayer(torch.nn.Module):
    """
    Completes the imputation via a learned feedforward net.
    It should be informative to explore it and see if the learning
    imputation is meaningful.
    I would guess not.
    """
    def __init__(
        self,
        time_steps: int,
        input_dimension: int
    ):
        """
        We need to initialize one layer that does a simple imputation.
        This imputation is done by reading the values from a matrix
        of size
        """
        super().__init__()
        self.imputer = torch.nn.Parameter(
            torch.randn(time_steps, input_dimension)
        )
        self.imputer.requires_grad = True

    def forward(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the imputation according to the following equation:

        imputed = X * (1-missing) + missing * (WX + b)

        where WX+b corresponds to the output of the linear layer and
        missing is a tensor such that the entries are 1 if the entry is
        missing in X and 0 otherwise. X is assumed to be
        filled such that missing entries are replaced with 0 by
        the longitudinal_phenomapping_dataset.LongitudinalPhenomappingDataset
        object.

        :param X: input tensor, shape = [batch_size, input_size]
        :param missing_mask: input tensor, shape = X.shape
        :return: imputed, shape = X.shape.
        """
        imputed = self.imputer * missing_mask
        imputed = X * (1-missing_mask) + imputed
        return imputed


class PeepholeLSTMCell(torch.nn.LSTMCell):
    """
    Implements the peephole LSTM method, where gates have access
    to the old cell state in the cell state update method.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        **kwargs
    ):
        """
        Initializing the peephole LSTM.
        :param input_size: int, dimension of the input time series.
        :param hidden_size: int, dimension of the hidden and cell state.
        :param bias: bool, include bias term or not?
        """
        super().__init__(input_size, hidden_size, bias, **kwargs)
        self.weight_ch = torch.nn.Parameter(
            torch.Tensor(3 * hidden_size, hidden_size)
        )
        if bias:
            self.bias_ch = torch.nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(hidden_size))
        self.reset_parameters()

    def forward(
        self,
        X: torch.Tensor,
        hx: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements forward method of the peephole LSTM.
        The update method is as follows:

        f_t = sigmoid(W_f * [C_{t-1}, h_{t-1}, x_{t}] + b_f)
        i_t = sigmoid(W_i * [C_{t-1}, h_{t-1}, x_{t}] + b_i)
        o_t = sigmoid(W_o * [C_{t}, h_{t-1}, x_{t}] + b_o)
        g_t = tanh(W_g * [C_{t}, h_{t-1}, x_{t}] + b_g)
        c_t = f_t haddamard c_{t-1} + i_t haddamard g_t
        h_t = o_t haddamard tanh(c_t)

        :param X: torch.Tensor.shape=[batch_size, input_size]
        :param hx: optional, tuple (h_t-1, c_t-1).
            h, hidden state = torch.Tensor.shape = [batch_size, hidden_size].
            Zero if not otherwise initialized.
            c, cell state = torch.Tensor.shape = [batch_size, hidden_size].
            Zero if not otherwise initialized.
        :return: (h_t, c_t). Updated hidden and cell state. Size for both is
            [batch_size, hidden_size].
        """
        if hx is None:
            zeros = torch.zeros(
                X.size(0), self.hidden_size, dtype=X.dtype, device=X.device
            )
            hx = (zeros.clone(), zeros.clone())
        h, c = hx

        wx = torch.nn.functional.linear(X, self.weight_ih, self.bias_ih)
        wh = torch.nn.functional.linear(h, self.weight_hh, self.bias_hh)
        wc = torch.nn.functional.linear(c, self.weight_ch, self.bias_ch)
        wxhc = wx + wh + torch.cat(
            (
                wc[:, :2 * self.hidden_size],
                torch.autograd.Variable(self.wc_blank).expand_as(h),
                wc[:, 2 * self.hidden_size:]
            ),
            1
        )
        i = torch.nn.functional.sigmoid(
            wxhc[:, :self.hidden_size]
        )
        f = torch.nn.functional.sigmoid(
            wxhc[:, self.hidden_size:2 * self.hidden_size]
        )
        g = torch.nn.functional.tanh(
            wxhc[:, 2*self.hidden_size:3*self.hidden_size]
        )
        o = torch.nn.functional.sigmoid(
            wxhc[:, 3*self.hidden_size:]
        )

        c = f * c + i * g
        h = o * torch.nn.functional.tanh(c)
        return (h, c)


class PositionalEmbeddingMatrix(torch.nn.Module):
    """
    Implements the positional embedding, as described
    in Vaiswani et. al. (2017). Offers the model
    information on the relative ordering of the
    tokens in the sequence.

    See: https://arxiv.org/abs/1706.03762 for info
    on default values and the meaning.
    """
    def __init__(
        self,
        d_model: int,
        n: int = 10000
    ):
        """
        Initialize arguments of the positional embedder.
        :param n: integer, default set to 10000. Controls
            the spacing between the fourier modes. I typically
            set this to be the size of the context window. Too
            small will cause t
        :param d_model: integer, specifying the number of
            dimensions in the embedding space.
        """
        super().__init__()

        self.n = n
        self.d_model = d_model

    def forward(
        self,
        ts: torch.Tensor
    ) -> torch.Tensor:
        """
        Creates and populates the positional embedding
        matrix, given the time values of the of the token
        data.

        :param ts: The time values of each token.
            ts.shape = [batch_size, seq_len]

        :return: positional_embedding_matrix, the evaluated
            values of times in the fourier basis.
            positional_embedding_matrix.shape =
                [batch_size, seq_len, d_model]
        """
        positional_embedding_matrix = torch.zeros(
            ts.shape[0], ts.shape[1], self.d_model
        )

        for k in range(self.d_model // 2):
            denom = self.n ** (2 * k / self.d_model)
            embed_even = torch.sin(ts / denom)
            embed_odd = torch.cos(ts / denom)
            positional_embedding_matrix[:, :, 2 * k] = embed_even
            positional_embedding_matrix[:, :, 2 * k + 1] = embed_odd

        return positional_embedding_matrix.to(device)


class AddNorm(torch.nn.Module):
    """
    Implements the residualization and normalization for
    the components in the transformer.
    """
    def __init__(
        self,
        dim_model: int
    ):
        """
        Initialized by including the normalized shape.

        :param dim_model: int, the dimension of the
        embedding space of the model
        """
        super().__init__()
        self.dim_model = dim_model
        self.norm = torch.nn.LayerNorm(self.dim_model).to(device)

    def forward(
        self,
        x: torch.Tensor,
        last_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the add normalization.

        :param x: Output of the attention module.
            x.shape = [batch_size, seq_len, dim_model]
        :param last_x: input to the attention module.
            last_x.shape = x.shape

        :return: normalized_tensor, the normalized output
            with a skip connection implemented.
        """
        return self.norm(x + last_x)


class Attention(torch.nn.Module):
    """
    Implements the attention module, assuming use of the
    scaled dot product attention model.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        d_hidden: int,
        padding_mask: torch.Tensor = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the forward pass of the model.

        Scaled Dot Product Attention is calculated as:
        Attention = SoftMax(Q K^{T}/sqrt(d_hidden)).
        Output = Attention * V.

        Note that the padding mask and the causal mask
        are applied before the softmax calculation.

        :param query: Queries of the attention triplet.
            query.shape = [batch_size, seq_len, d_hidden]
        :param key: Keys of the attention triplet.
            key.shape = [batch_size, seq_len, d_hidden]
        :param value: Values of the attention triplet.
            value.shape = [batch_size, seq_len, d_hidden]
        :param d_hidden: int, Dimension of attention space.
        :param padding_mask: torch.Tensor. Masks out any
            attention paid to the pad tokens.
            padding_mask.shape = [batch_size, seq_len]

        :return:
            output: Output of the attention calculation.
                output.shape = [batch_size, seq_len, d_hidden]
            attention: Attention weights from the attention calculation.
                attention.shape = [batch_size, seq_len, d_hidden]
        """
        scores = torch.matmul(
            query, key.transpose(-1, -2)
        )/torch.sqrt(torch.tensor(d_hidden))

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-1)
            mask = torch.matmul(
                (1 - padding_mask),
                (1 - padding_mask).transpose(-1, -2)
            )
            # Mask should be shaped [batch_size, seq len, seq len].
            # Ones should be in the upper block where are non pad, 0 elsewhere.
            mask = mask.unsqueeze(1)
            # Unsqueeze in dimension 1 for broadcasting over multiple heads.
            scores -= (1 - mask)*1e9
        attention = torch.nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attention, value)
        return attention, output


class MultiHeadAttention(torch.nn.Module):
    """
    Implements multiheaded attention for arbitrary
    attention style.
    """
    def __init__(
        self,
        heads: int,
        d_query: int,
        d_key: int,
        d_value: int,
        d_hidden: int,
        d_model: int,
        attention: torch.nn.Module = Attention()
    ):
        """
        Initializing the mutlihead attention module. Creates
        feedforward layers for the (Q, K, V) triplets for each
        head of the attention, and then a projection layer
        at the end to project it back into the space of the model.

        :param heads: int, the number of heads for multihead attention.
        :param d_query: int, the input dimension size of the entry to query.
        :param d_key: int, the input dimension of the keys.
        :param d_value: int, the input dimension of the values.
        :param d_hidden: int, the size of the embedding space of the attention
            mechanism.
        :param d_model: int, the final dimension of hte attention output.
        :param attention: torch.nn.Module, the style of attention to use.
            Defaults to scaled dot product attention.
        """
        super().__init__()
        self.heads = heads
        self.d_query = d_query
        self.d_key = d_key
        self.d_value = d_value
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.attention = attention

        self.W_q = torch.nn.Linear(
            self.d_query, self.heads * self.d_hidden
        ).to(device)
        self.W_k = torch.nn.Linear(
            self.d_key, self.heads * self.d_hidden
        ).to(device)
        self.W_v = torch.nn.Linear(
            self.d_value, self.heads * self.d_hidden
        ).to(device)

        self.W_o = torch.nn.Linear(
            self.heads * self.d_hidden, self.d_model
        ).to(device)

    def reshape_tensor(
        self,
        x: torch.Tensor,
        flag: bool
    ) -> torch.Tensor:
        """
        Reshaping Tensor into shape that flattens over heads or folds over
        heads.

        :param x: torch.Tensor. Is either in shape:
            x.shape = [batch_size, seq_len, heads * d_hidden]
            OR
            x.shape = [batch_size, heads, seq_len, d_hidden]
        :flag bool: Tells us which type of x shape to expect.
            If true, we expect:
                x.shape = [batch_size, seq_len, heads * d_hidden]
            else:
                x.shape = [batch_size, heads, seq_len, d_hidden]

        :return: reshaped_tensor.
            If flag == True:
                reshaped_tensor.shape = [batch_size, heads, seq_len, d_hidden]
            else:
                reshaped_tensor.shape = [batch_size, seq_len, d_hidden]
        """
        if flag:
            # Convert from shape [batch_size, seq_len, heads * d_hidden]
            # to [batch_size, heads, seq_len, d_hidden]
            x = x.reshape(
                x.shape[0], x.shape[1], self.heads, x.shape[2]//self.heads
            )
            # x.shape = [batch_size, seq_len, heads, d_hidden]
            x = x.permute(0, 2, 1, 3)
        else:
            # Convert from shape [batch_size, heads, seq_len, d_hidden]
            # to shape [batch_size, seq_len, heads*d_hidden]
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the forward pass of the multihead attention module.

        :param query: Query of the QKV triplet.
            query.shape = [batch_size, seq_len, d_query]
        :param key: Key of the QKV triplet.
            key.shape = [batch_size, seq_len, d_key]
        :param value: Value of the QKV triplet.
            value.shape = [batch_size, seq_len, d_value]
        :param padding_mask: tensor that is 1 where the input has a
            pad token and 0 elsewhere.
            padding_mask.shape = [batch_size, seq_len]

        :return: Output and attention values of MHA.
            output.shape = [batch_size, seq_len, d_model]
            attention.shape = [batch_size, heads, seq_len, seq_len]
        """
        q, k, v = self.W_q(query), self.W_k(key), self.W_v(value)

        q = self.reshape_tensor(q, True)
        k = self.reshape_tensor(k, True)
        v = self.reshape_tensor(v, True)

        attention, output = self.attention(
            q,
            k,
            v,
            self.d_hidden,
            padding_mask
        )

        output = self.reshape_tensor(output, False)
        output = self.W_o(output)
        return output, attention

import torch
import pandas as pd
import numpy as np
import typing
from collections import defaultdict

import patient_feature_learning.constants as ct

IDX_TIME_SEGMENT_NUMBER = 0
IDX_DIAGNOSIS_CODES = 1
IDX_MEDICATIONS_CODES = 2
IDX_MEASUREMENTS = 3
IDX_LABS = 4

IDXs = {
    'diagnosis': IDX_DIAGNOSIS_CODES,
    'medications': IDX_MEDICATIONS_CODES,
    'measurements': IDX_MEASUREMENTS,
    'labs': IDX_LABS
}


def _get_num_quantiles(
    quantiles: pd.DataFrame
) -> int:
    """
    Identifies the number of quantiles
    specified by a slice of the quantiles
    dataframe. Handles the case where some
    quantiles aren't present, and the binary
    case.
    :param quantiles: slice of the quantiles dataframe
        corresponding to a particular index
    :return: num_quantiles
    """
    num_quantiles = max(quantiles)

    if np.isnan(num_quantiles):
        num_quantiles = 0
    else:
        num_quantiles = int(num_quantiles)

    return num_quantiles


class LongitudinalPhenomappingDataset(torch.utils.data.Dataset):
    """
    torch.utils.data.Dataset object that creates dictionary with
    the keys for longitudinal phenomapping. Takes in a collection
    of patients with a series of visits and generates a dictionary
    with a tensor representation of the visits, masking for
    visits that did not occur, and a set of demographics.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_length: int,
        code_indices: typing.Dict[str, typing.Tuple[str, ...]],
        quantile_labs: pd.DataFrame,
        quantile_measurements: pd.DataFrame,
        use_age: bool,
        use_sex: bool,
        saved_keys: typing.Optional[typing.List] = None,
        targets: typing.Optional[typing.List] = [],
        time_segments_per_year: int = 4
    ):
        """
        Initializes the dataset object.

        :param dataframe: pandas.Dataframe object with
            subject and visits data.
        :param time_length: The number of visits over the lookback
            window being considered.
        :param code_indices: Will be used to map the
            code to the correct index vector.
        :param quantile_labs: Dataframe that has the
                number of quantiles and the ranges for labs.
        :param quantile_measurements: Dataframe that has the
                number of quantiles and ranges for measurements.
        :param use_age: bool, Decides whether or not we use age.
        :param use_sex: bool, Decides whether or not we use sex.
        :param saved_keys: List of keys corresponding to which
            medical codes to use for tensor representation.
        :param targets: List of column names in dataframe that are
            targets for supervision.
        :param default_day:
        """

        self.dataframe = dataframe
        self.time_length = time_length
        self.code_indices = code_indices
        self.quantile_labs = quantile_labs
        self.quantile_measurements = quantile_measurements
        self.targets = targets
        self.time_segments_per_year = time_segments_per_year

        for target in targets:
            self.dataframe[target] = self.dataframe[target].astype(float)

        self.use_age = use_age
        self.use_sex = use_sex

        self.index_translation = self._get_indices()

        if saved_keys is not None:
            self.saved_keys = defaultdict(int)
            for key in saved_keys:
                self.saved_keys[key] = 1
        else:
            self.saved_keys = defaultdict(int)
            for key in self.code_indices.keys():
                for code in self.code_indices[key]:
                    code_name = code + '_' + key
                    self.saved_keys[code_name] = 1

        self.n_combined_indicators, self.medical_code_to_index = (
            self.get_medical_code_to_index()
        )

        # Create assertions to ensure the dataframe has the right structure.
        if ("day0" not in self.dataframe.columns):
            raise KeyError("day0 not provided in dataframe")

        for target in self.targets:
            if target not in self.dataframe.columns:
                raise KeyError(f'{target} column not provided in dataframe')

        if len(
            set(self.targets).intersection(
                set(['X', 'missing_mask', 'demographics'])
            )
        ) > 0:
            raise ValueError("An entry in target overwrites critical keys")

    def get_medical_code_to_index(
        self
    ) -> (int, typing.Dict):
        """
        Translates medical code name into an index
        value for the position in the tensor
        :return: medical_code_to_index
        :return: index_to_start, number of combined indicators in
            dataset.
        """
        code_idx_to_dim_tx = {}
        index_to_start = 0

        quantiles_dict = {
            'measurements': self.quantile_measurements,
            'labs': self.quantile_labs
        }

        for code in ['diagnosis', 'medications']:
            for i in self.code_indices[code]:
                code_name = i + '_' + code
                if self.saved_keys[code_name] == 1:
                    code_idx_to_dim_tx[code_name] = index_to_start
                    index_to_start += 1
        for code in ['measurements', 'labs']:
            for idx, val in enumerate(self.code_indices[code]):
                code_name = val + '_' + code
                if self.saved_keys[code_name] == 1:
                    nm_quantiles = _get_num_quantiles(
                        quantiles_dict[code][
                            quantiles_dict[
                                code
                            ]['medical_code_idx'] == float(idx)
                        ]['quantile']
                    )
                    if nm_quantiles > 0:
                        code_idx_to_dim_tx[code_name] = [
                            index_to_start + i for i in range(nm_quantiles)
                        ]
                        index_to_start += nm_quantiles
                    else:
                        code_idx_to_dim_tx[code_name] = index_to_start
                        index_to_start += 1
        return index_to_start, code_idx_to_dim_tx

    def __len__(self):
        return len(self.dataframe)

    def _get_indices(
        self
    ) -> typing.List[typing.Tuple[int, int]]:
        """
        Creates (subject, visit) pairs, where the
        subject index corresponds to a row in the
        processed dataframe and the visit index is
        one visit from that row (array in column
        'visit_data').
        :return: index_translation
        """
        index_translation = []
        subject_index = 0

        for subject_id, visits in zip(
            self.dataframe['subject_id'], self.dataframe['visits_data']
        ):
            for visit_index in range(len(visits)):
                index_translation.append((subject_index, visit_index))
            subject_index += 1
        return index_translation

    def _get_demographics(
        self,
        index: int
    ) -> typing.Optional[torch.Tensor]:
        """
        Obtains the age and sex of the patient, with
        reference to an index date. Currently defaults
        to Jan 1, 1915.

        :param index: Indexes which patient to get age
            and sex of.
        :return: age_sex tensor.
        """

        subject_index, visit_index = self.index_translation[index]
        day0 = pd.Timestamp(self.dataframe['day0'][subject_index])

        if self.use_age:
            time_segment_number = self.dataframe['visits_data'][
                subject_index
            ][visit_index]['time_segment_number']
            age = (
                day0
                + pd.Timedelta(
                    days=(
                        ct.days_per_year
                        * time_segment_number
                        / self.time_segments_per_year
                    )
                )
                - pd.Timestamp(self.dataframe['birth_date'][subject_index])
            ).total_seconds() / (
                ct.days_per_year
                * ct.hours_per_day
                * ct.minutes_per_hour
                * ct.seconds_per_minute
            )

        if self.use_sex:
            sex = float(self.dataframe['sex'][subject_index].upper()[0] == 'F')

        if self.use_age and self.use_sex:
            return torch.Tensor([age, sex])
        elif self.use_age and not self.use_sex:
            return torch.Tensor([age])
        elif self.use_sex and not self.use_age:
            return torch.Tensor([sex])
        else:
            return torch.Tensor([0])

    def _translate_to_vecs(
        self,
        medical_codes: torch.Tensor,
        medical_mask: torch.Tensor,
        visits_data: typing.Dict
    ) -> (torch.Tensor, torch.Tensor):
        """
        Populates a vector of all 0s with a 1
        if that particular code or quantile
        appears in a given visit. If result
        is missing, it is marked in the mask.

        :param medical_codes: A vector of all 0s.
            For the code representation.
        :param medical_mask: A vector of all 0s.
            For the mask.
        :param visits_data: A dictionary containing
            which codes occurred in a given visit.

        :return: medical_codes, A vector populated with
            1s corresponding to which codes occurred in a visit.
        :return: medical_mask, A vector populated with 1s corresponding
            to which codes had missing results.
        """
        for key in visits_data.keys():
            if key in ['diagnosis_codes', 'medications_codes']:
                type_of_key = key.split('_')[0]
                if visits_data[key] is not None:
                    for val in visits_data[key]:
                        code_name = (
                            self.code_indices[type_of_key][int(val)]
                            + '_' 
                            + type_of_key
                        )
                        if self.saved_keys[code_name] == 1:
                            medical_codes[
                                self.medical_code_to_index[code_name]
                            ] = 1.
            elif key in ['measurements_numeric', 'labs_numeric']:
                type_of_key = key.split('_')[0]
                if visits_data[key] is not None:
                    for val in visits_data[key]:
                        code_name = (
                            self.code_indices[type_of_key][
                                int(val['medical_code_idx'])
                            + '_' 
                            + type_of_key
                        )
                        if self.saved_keys[code_name] == 1:
                            if len(
                                val['quantiles'] > 0
                            ) and isinstance(
                                self.medical_code_to_index[code_name], list
                            ):
                                for quantile in val['quantiles']:
                                    medical_codes[
                                        self.medical_code_to_index[
                                            code_name
                                        ][quantile - 1]
                                    ] = 1.
                            elif len(
                                val['quantiles'] == 0
                            ) and isinstance(
                                self.medical_code_to_index[code_name], list
                            ):
                                medical_mask[
                                    self.medical_code_to_index[
                                        code_name
                                    ]
                                ] = 1.
                            else:
                                medical_codes[
                                    self.medical_code_to_index[
                                        code_name
                                    ]
                                ] = 1.
        return medical_codes, medical_mask

    def _construct_visit_vec(
        self,
        index: int,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Constructs the vector corresponding to
        patient, visit pairs, indexed by
        self.index_translation.

        :param index: Identifies which
            patient, visit pair to return
            tensors for.

        :return: medical_vec, tensor representation
            of visit.
        :return: medical_mask, tensor representation
            of any missing entry in visit.
        """
        subject_index, visit_index = self.index_translation[index]
        visits_data = (
            self.dataframe['visits_data']
            [subject_index]
            [visit_index]
        )

        medical_vec = torch.zeros(
            self.n_combined_indicators,
            dtype=torch.float
        )

        medical_mask = torch.zeros(
            self.n_combined_indicators,
            dtype=torch.float
        )

        self._translate_to_vecs(
            medical_vec,
            medical_mask,
            visits_data
        )

        return medical_vec, medical_mask

    def __getitem__(
        self,
        index
    ) -> typing.Dict:
        """
        Gets the demographics, visit representation, and mask
        for the indexed patient.

        :param index: Index of patient to get data for.
        :return: dictionary of the three targets
            dictionary keys:
                X: Binary tensor representation of visit
                    trajectory.
                    X.shape = [time_length, n_combined_indicators].
                missing_mask: Binary representation of missing visits.
                    X.shape = missing_mask.shape
                Demos: A two entry tensor, with the age and sex of the
                    patient at the beginning of the trajectory.
        """
        X = torch.zeros((self.time_length, self.n_combined_indicators))
        missing_mask = torch.zeros_like(X)

        visits_data = self.dataframe['visits_data'][index]

        min_time_segment_number = visits_data[0]['time_segment_number']

        indices_hit = []

        for visit in visits_data:
            time_segment_number = visit['time_segment_number']
            idx_to_update = time_segment_number - min_time_segment_number
            indices_hit.append(idx_to_update)

            (
                X[idx_to_update],
                missing_mask[idx_to_update]
            ) = self._translate_to_vecs(
                X[idx_to_update],
                missing_mask[idx_to_update],
                visit
            )

        for i in range(self.time_length):
            if i not in indices_hit:
                missing_mask[i] = 1.

        counter = 0
        for elem in self.index_translation:
            if elem[0] == index and elem[1] == 0:
                demos = self._get_demographics(counter)
                break
            else:
                counter += 1

        elem = {'X': X, 'missing_mask': missing_mask, 'demographics': demos}

        for target in self.targets:
            elem[target] = torch.tensor(self.dataframe[target][index])

        return elem


class MaskedByVisitDataset(LongitudinalPhenomappingDataset):
    """
    Applies masking to tensors in the visit representation.
    What is meant by this is that given a patient medical record,
    represented as an (num codes x num visits) representation,
    random visits are replaced with columns of all 0s. The model
    will then be expected to recover which codes were anticipated
    to have occurred in that visit.
    :class:~patient_feature_learning.LongitudinalPhenomappingDataset
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_length: int,
        code_indices: typing.Dict[str, typing.Tuple[str, ...]],
        quantile_labs: pd.DataFrame,
        quantile_measurements: pd.DataFrame,
        use_age: bool,
        use_sex: bool,
        saved_keys: typing.Optional[typing.List] = None,
        targets: typing.Optional[typing.List] = None,
        time_segments_per_year: int = 4,
        masking_fraction: float = 0.15
    ):
        """
        See documentation for LongitudinalPhenomappingDataset
        to see most of the descriptions.

        See :class:.LongitudinalPhenomappingDataset.

        :param masking_fraction: The average number of visits
            to be replaced with columns of 0s.
        """
        super().__init__(
            dataframe,
            time_length,
            code_indices,
            quantile_labs,
            quantile_measurements,
            use_age,
            use_sex,
            saved_keys,
            targets
        )
        self.masking_fraction = masking_fraction

    def __getitem__(
        self,
        index
    ) -> typing.Dict:
        """
        Gets item, using most of the behavior from the parent class.
        Masking is done after that.
        The returned dictionary contains one additional key compared to
        the parent class's method.
            key:
                masked_X: A masked representation of the visit trajectory,
                    where masked visits are replaced with all 0s.
                    masked_X.shape = X.shape.
        """
        elem = super().__getitem__(index)
        masked_X = elem['X'].clone()

        for i in range(self.time_length):
            if torch.rand(1) < self.masking_fraction:
                masked_X[i] = 0.

        elem['masked_X'] = masked_X
        return elem


class TokenizedRepresentation(LongitudinalPhenomappingDataset):
    """
    Represents data as a series of codes, in order of when they
    appeared in the trajectory, up to a total number of some
    context length of codes. This is more traditional for how
    NLP data is represented.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_length: int,
        code_indices: typing.Dict[str, typing.Tuple[str, ...]],
        quantile_labs: pd.DataFrame,
        quantile_measurements: pd.DataFrame,
        use_age: bool,
        use_sex: bool,
        saved_keys: typing.Optional[typing.List] = None,
        targets: typing.Optional[typing.List] = None,
        time_segments_per_year: int = 4,
        spec_tokens: typing.List = ['CLS', 'SEP', 'MASK', 'EOV', 'PAD'],
        max_seq_length: int = 2048,
        masking_fraction: float = 0.125,
        mutating_fraction: float = 0.025
    ):
        """
        Initializes the class. Most parameters are described in
        LongitudinalPhenomappingDataset.
        See :class:.LongitudinalPhenomappingDataset.

        :param spec_tokens: List of special tokens that need to
            be represented. Includes thinks like start of sequence,
            separation between visits, mask tokens that need to be predicted,
            end of visit tokens, and padding tokens.
        :param max_seq_length: Int representing length of the context window.
        :param masking_fraction: Number of non-special tokens to be masked.
        :param mutation_fraction: Number of non-special tokens to be mutated.
        """
        super().__init__(
            dataframe,
            time_length,
            code_indices,
            quantile_labs,
            quantile_measurements,
            use_age,
            use_sex,
            saved_keys,
            targets,
            time_segments_per_year
        )
        self.masking_fraction = masking_fraction
        self.mutating_fraction = mutating_fraction
        self.max_seq_length = max_seq_length
        self.spec_tokens = spec_tokens

        for index, token in enumerate(spec_tokens):
            self.medical_code_to_index[token] = (
                self.n_combined_indicators + index
            )

    def _get_tokenized_rep(
        self,
        index: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Takes in a subject index and converts their visit data into a tokenized
        representation.

        :param index: The index of the subject in the dataframe.

        :return: tokenized_rep, torch.Tensor of shape [max_seq_length].
            Each entry corresponds to the index of the code that appears
            at that moment.
        :return: times, torch.Tensor of shape [max_seq_length].
            Each entry corresponds to the age of the patient when a
            particular code appears.
        :return: attention_mask, torch.Tensor of shape [max_seq_length].
            Is 1 where pad token appears, 0 elsewhere.
        """

        def add_token(
            tokenized_rep: torch.Tensor,
            ages: torch.Tensor,
            times: torch.Tensor,
            curr_token_index: int,
            code: int,
            age: float,
            time: int,
        ) -> (torch.Tensor, torch.Tensor, torch.Tensor, int):
            """
            Temporary function for adding tokens to our representation
            sequentially.

            :param tokenized_rep: The tokenized representation thus far.
            :param ages: The ages thus far.
            :param times: The visit segment number thus far.
            :param curr_token_index: The index at which we add the token.
            :param code: The token we are adding.
            :param age: The age we are adding.
            :param time: The time we are adding.

            :return:
                tokenized_rep: Updated tokenized representation.
                ages: Updated ages.
                times: Updated times.
                curr_token_index: Updated token index.
            """
            tokenized_rep[curr_token_index] = code
            ages[curr_token_index] = age
            times[curr_token_index] = time
            curr_token_index += 1
            return tokenized_rep, ages, times, curr_token_index

        day0 = pd.Timestamp(self.dataframe['day0'][index])

        tokenized_rep = (
            self.medical_code_to_index['PAD'] * torch.ones(
                self.max_seq_length, dtype=torch.int
            )
        )
        times = torch.zeros(self.max_seq_length, dtype=torch.float)
        ages = torch.zeros(self.max_seq_length, dtype=torch.float)

        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.float)

        # Initialize start of sequence token.
        tokenized_rep[0] = self.medical_code_to_index['CLS']

        curr_token_index = 1

        subjects_visits_data = self.dataframe['visits_data'][index]
        min_time_segment_number = (
            subjects_visits_data[0]['time_segment_number']
        )

        for visit in subjects_visits_data:
            if curr_token_index < self.max_seq_length:
                time_segment_number = visit['time_segment_number']
                age = (
                    day0
                    + pd.Timedelta(
                        days=(
                            ct.days_per_year
                            * time_segment_number
                            / self.time_segments_per_year
                        )
                    )
                    - pd.Timestamp(self.dataframe['birth_date'][index])
                ).total_seconds() / (
                    ct.days_per_year
                    * ct.hours_per_day
                    * ct.minutes_per_hour
                    * ct.seconds_per_minute
                )
                time = time_segment_number - min_time_segment_number

                for code_type in ['diagnosis_codes', 'medications_codes']:
                    if visit[code_type] is not None:
                        for code in visit[code_type]:

                            code_type_str = code_type.split('_')[0]
                            code_name = (
                                self.code_indices[code_type_str][int(code)]
                                + '_'
                                + code_type_str
                            )

                            if self.saved_keys[code_name] == 1:
                                code = self.medical_code_to_index[code_name]
                                (
                                    tokenized_rep,
                                    ages,
                                    times,
                                    curr_token_index
                                ) = add_token(
                                    tokenized_rep,
                                    ages,
                                    times,
                                    curr_token_index,
                                    code,
                                    age,
                                    time
                                )

                for code_type in ['measurements_numeric', 'labs_numeric']:
                    if visit[code_type] is not None:
                        for code in visit[code_type]:

                            code_type_str = code_type.split('_')[0]
                            code_number = int(code['medical_code_idx'])
                            code_name = (
                                self.code_indices[code_type_str][code_number]
                                + '_'
                                + code_type_str
                            )

                            if self.saved_keys[code_name] == 1:
                                # TODO: Handle cases where lab reported but
                                # no result is given.
                                if (
                                    len(code['quantiles']) == 0
                                    and not isinstance(
                                        self.medical_code_to_index[code_name],
                                        list
                                    )
                                ):
                                    code = self.medical_code_to_index[
                                        code_name
                                    ]
                                    (
                                        tokenized_rep,
                                        ages,
                                        times,
                                        curr_token_index
                                    ) = add_token(
                                        tokenized_rep,
                                        ages,
                                        times,
                                        curr_token_index,
                                        code,
                                        age,
                                        time
                                    )
                                else:
                                    for quantile in code['quantiles']:
                                        code = self.medical_code_to_index[
                                            code_name
                                        ][quantile - 1]
                                        (
                                            tokenized_rep,
                                            ages,
                                            times,
                                            curr_token_index
                                        ) = add_token(
                                            tokenized_rep,
                                            ages,
                                            times,
                                            curr_token_index,
                                            code,
                                            age,
                                            time
                                        )
                (
                    tokenized_rep,
                    ages,
                    times,
                    curr_token_index
                ) = add_token(
                    tokenized_rep,
                    ages,
                    times,
                    curr_token_index,
                    self.medical_code_to_index['SEP'],
                    age,
                    time
                )
        if curr_token_index < self.max_seq_length:
            (
                tokenized_rep,
                ages,
                times,
                curr_token_index
            ) = add_token(
                tokenized_rep,
                ages,
                times,
                curr_token_index,
                self.medical_code_to_index['EOV'],
                age,
                time
            )

        # Fill in the rest with PAD and tell the padding mask that.
        attention_mask[curr_token_index:] = 1.

        return tokenized_rep, ages, times, attention_mask

    def _masker(
        self,
        tokenized_rep: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the masking and mutation for traditional
        MLM tasks.

        :param tokenized_rep: A representation of a visit
            trajectory.
            tokenized_rep.shape = [max_seq_length]
        :param attention_mask: A representation of the
            padding tokens. 1 where pads exist.
            attention_mask.shape = tokenized_rep.shape.

        :return: masked_tokenized_rep. A tensor where
            some tokens are replaced with mask tokens
            and others are mutated. Only non-special
            tokens can be masked/mutated.
            masked_tokenized_rep.shape = tokenized_rep.shape
        """
        num_filled_tokens = self.max_seq_length - torch.sum(attention_mask)
        tokens_to_edit = torch.bernoulli(
            (
                (self.masking_fraction + self.mutating_fraction)
                * torch.ones(int(num_filled_tokens))
            )
        )

        mask_or_mutate = torch.where(tokens_to_edit == 1)[0]
        number_to_mutate = len(mask_or_mutate)

        relative_fraction = self.mutating_fraction / (
            self.masking_fraction
            + self.mutating_fraction
        )

        mutating = mask_or_mutate[
            torch.randperm(
                number_to_mutate
            )[:int(relative_fraction * number_to_mutate)]
        ]

        masked_tokenized_rep = tokenized_rep.clone()

        for index in mask_or_mutate:
            if (
                index != 0
                and index != (num_filled_tokens - 1)
                and tokenized_rep[index] != self.medical_code_to_index['SEP']
            ):
                # Don't mutate start, end, or separator tokens.
                if index in mutating:
                    # Mutate to any non-special token.
                    mutate_to = torch.randint(
                        low=0, high=self.n_combined_indicators, size=(1,)
                    )
                    masked_tokenized_rep[index] = mutate_to
                else:
                    masked_tokenized_rep[index] = (
                        self.medical_code_to_index['MASK']
                    )
        return masked_tokenized_rep

    def _get_target_distro(
        self,
        tokenized_rep: torch.Tensor
    ) -> torch.Tensor:
        """
        Obtains what should be expected from the unmasker, given
        what the initial tokenized rep is.

        :param tokenized_rep: torch.Tensor, representing the sequence
            of codes at each visit.
            tokenized_rep.shape = [max_seq_length]

        :return: target_rep, torch.Tensor, a binary matrix with 1
            if a particular token appears at that site and 0 otherwise.
            target_rep.shape = [max_seq_length, num_tokens]
        """

        # Total number of tokens is n_combined_indicators + number of
        # spec tokens.
        num_tokens = self.n_combined_indicators + len(self.spec_tokens)

        target_rep = torch.zeros((self.max_seq_length, num_tokens))

        for i in range(self.max_seq_length):
            target_rep[i, tokenized_rep[i]] = 1.

        return target_rep

    def __getitem__(
        self,
        index: int
    ) -> typing.Dict:
        """
        Return a dictionary with all the needed inputs
        for training an MLM-based model.

        :param index: The subject index we are tokenizing.

        :return: Dictionary with keys needed to train an MLM.
            keys:
                tokens: Tensor containing the index value of all the tokens
                    that appear in a visit trajectory, up to the context
                    window length. This will be used during inference.
                masked_tokens: Tensor with random masks and mutations. This
                    will be used during training.
                ages: Tensor containing the age of the patient at each visit.
                times: Tensor containing the time from the beginning of the
                    lookback window.
                padding_mask: Binary tensor identifying which tokens are pad
                    tokens.
                target_prediction: The target values for the
                    unmasker.
        """
        tokens, ages, ts, padding_mask = self._get_tokenized_rep(
            index
        )
        masked_tokens = self._masker(tokens, padding_mask)
        target_rep = self._get_target_distro(tokens)
        elem = {
            'tokens': tokens,
            'masked_tokens': masked_tokens,
            'ages': ages,
            'times': ts,
            'padding_mask': padding_mask,
            'target_prediction': target_rep,
        }
        for target in self.targets:
            elem[target] = torch.tensor(self.dataframe[target][index])

        if self.dataframe['sex'][index].upper()[0] == 'M':
            sex = torch.tensor(0.)
        else:
            sex = torch.tensor(1.)
        elem['sex'] = sex
        return elem

import torch
import pandas as pd
import pyarrow.parquet
import numpy as np
import typing
import random
from torch.utils.data._utils.collate import default_collate

# TODO: Define these numbers near where the original parquet file is generated.
IDX_TIME_SEGMENT_NUMBER = 0
IDX_DIAGNOSIS_CODES = 1
IDX_MEDICATIONS_CODES = 2
IDX_MEASUREMENTS = 3
IDX_LABS = 4

IDXs = {
    'diagnosis': IDX_DIAGNOSIS_CODES,
    'medications': IDX_MEDICATIONS_CODES,
    'measurements': IDX_MEASUREMENTS,
    'labs': IDX_LABS
}


def _translate_to_vecs(
    medical_codes: torch.Tensor,
    visits_data: pyarrow.StructScalar,
    code_indices: typing.Dict[str, str],
    medical_code_to_index: typing.Dict[str, int]
) -> torch.Tensor:
    """
    Populates a vector of all 0s with a 1
    if that medical code (or the particular quantile)
    appears in a given visit
    :param medical_codes: empty initial vector
    :param visits_data: a structscalar corresponding
        to a particular visit
    :return: medical_codes
    """
    for key, value in IDXs.items():
        if key in ['diagnosis', 'medications']:
            if visits_data[value].as_py() is not None:
                for val in visits_data[value].as_py():
                    code_name = code_indices[key][int(val)]
                    medical_codes[medical_code_to_index[code_name]] = 1.
        else:
            if visits_data[value].as_py() is not None:
                for val in visits_data[value].as_py():
                    code_name = code_indices[key][int(
                        val['medical_code_idx'])]
                    if len(val['quantiles']) > 0:
                        for quantile in val['quantiles']:
                            medical_codes[
                                medical_code_to_index[code_name][quantile-1]
                            ] = 1.
                    else:
                        medical_codes[medical_code_to_index[code_name]] = 1.

    return medical_codes


def _get_num_quantiles(
    quantiles: pd.DataFrame
) -> int:
    """
    Identifies the number of quantiles
    specified by a slice of the quantiles
    dataframe. Handles the case where some
    quantiles aren't present, and the binary
    case.
    :param quantiles: slice of the quantiles dataframe
        corresponding to a particular index
    :return: num_quantiles
    """
    num_quantiles = max(quantiles)

    if np.isnan(num_quantiles):
        num_quantiles = 0
    else:
        num_quantiles = int(num_quantiles)

    return num_quantiles


def patient_collate(
    tensor_dict_list: typing.List[
        typing.Dict[str, torch.Tensor]
    ]
) -> typing.Dict[str, torch.Tensor]:
    """
    A custom collate function that will
    behave as default for most of the entries.
    It takes special care to manage the co-occurrence
    pairs from negative and positive sampling, in order
    to deal with the fact that they are not guaranteed
    to have the same shape across all samples.

    :param tensor_dict_list: List of the outputs of
        __getitem__ from med2vec_dataset.ArrowDataset
        object.
    :return: A dictionary of the batched output
    """

    # Tensor dict list will have a list of
    # dictionaries each example from the
    # dataset object.

    collated_tensors = {}

    # Iterate over keys, collating with specific logic
    # for the cooc pairs cases, because that's where
    # tensors of different sizes can show up.

    for key in tensor_dict_list[0].keys():
        tensors = [
            tensor_dict[key]
            for tensor_dict
            in tensor_dict_list
            if tensor_dict[key] is not None
        ]
        if key in ["pos_cooc_pairs", "neg_cooc_pairs"]:
            # If it's a cooc pairs, we just concatenate over the
            # 0th dimension. Note that all examples in the tensors
            # list for this case will be [pairs, 2].
            collated_tensors[key] = torch.cat(tensors)
        else:
            # Everything else can be treated without any special
            # sauce.
            collated_tensors[key] = default_collate(tensors)

    return collated_tensors


def get_medical_code_to_index(
    code_indices: dict,
    quantiles_labs: pd.DataFrame,
    quantiles_measurements: pd.DataFrame
) -> (int, typing.Dict[str, int]):
    """
    Creates a neat little dictionary to quickly map
    from the code index to the column index of the vector
    we are encoding.

    :param code_indices: generated by preprocessing.preprocess.
            It yields index in the preprocess framework to name
    :param quantiles_labs: Dataframe that maps quantile number to
        medical code index and the min/max bound for a quantile
    :param quantiles_measurements: same as above, but for measurements
    :return: code_idx_to_dim_tx
    """
    code_idx_to_dim_tx = {}
    index_to_start = 0

    quantiles_dict = {
        'measurements': quantiles_measurements,
        'labs': quantiles_labs
    }

    code_types = ['diagnosis', 'medications', 'measurements', 'labs']

    for code in code_types:
        if code in ['diagnosis', 'medications']:
            for i in code_indices[code]:
                code_idx_to_dim_tx[i] = index_to_start
                index_to_start += 1

        else:
            for idx, val in enumerate(code_indices[code]):
                num_quantiles = _get_num_quantiles(
                    quantiles_dict[code][
                        quantiles_dict[code]['medical_code_idx'] == float(idx)
                    ]['quantile']
                )
                if num_quantiles > 0:
                    code_idx_to_dim_tx[val] = [
                        index_to_start + i for i in range(num_quantiles)
                    ]
                    index_to_start += num_quantiles
                else:
                    code_idx_to_dim_tx[val] = index_to_start
                    index_to_start += 1

    return index_to_start, code_idx_to_dim_tx


class ArrowDataset(torch.utils.data.Dataset):
    """
    Torch dataset for constructing the x, y pairs for med2vec.
    Indices are a little messy but we can break each index into two pieces:
    (patient index, time segment index).
    """
    def __init__(
        self,
        dataframe: pyarrow.Table,
        window_size: int,
        code_indices: typing.Dict[str, typing.Tuple[str, ...]],
        quantile_labs: pd.DataFrame,
        quantile_measurements: pd.DataFrame,
        max_occupancy: int,
        use_age: bool,
        use_sex: bool,
    ):
        """
        Takes in the dataset from the preprocessed PFL framework
        and can be used to slice through the dataset.

        :param dataframe: the dataframe with patient+visit data
        :param window_size: the window size for prediction in med2vec
        :param code_indices: will be used to map the
            code to the correct index vector
        :param quantile_labs: dataframe that has the
                number of quantiles and the ranges for labs
        :param quantile_measurements: dataframe that has the
                number of quantiles and ranges for measurements
        :param max_occupancy: max occupancy of binary
                vector for neg samples
        :param use_age: bool, decides whether or not we use age
        :param use_sex: bool, decides whether or not we use sex
        """
        self.dataframe = dataframe
        self.window_size = window_size
        self.code_indices = code_indices
        self.quantile_labs = quantile_labs
        self.quantile_measurements = quantile_measurements

        self.max_occupancy = max_occupancy

        self.use_sex = use_sex
        self.use_age = use_age

        self.index_translation = self._get_indices()

        self.n_combined_indicators, self.medical_code_to_index = (
            get_medical_code_to_index(
                self.code_indices,
                self.quantile_labs,
                self.quantile_measurements
            )
        )

    def __len__(self):
        """
        Returns the number overall entries in the dataframe.
        :return: total number of visits
        """
        return len(self.index_translation)

    def _get_indices(
        self,
    ) -> typing.List[typing.Tuple[int, int]]:
        """
        Creates (subject, visit) pairs, where the
        subject index corresponds to a row in the
        preprocessed dataframe, and the visit index
        is one visit from that row (array in column
        'visit data').
        :return: index_translation
        """
        index_translation = []
        subject_index = 0
        dataframe_index = 0
        for subject_id, visits in zip(
            self.dataframe['subject_id'], self.dataframe['visits_data']
        ):
            for visit_index in range(len(visits)):
                index_translation.append((subject_index, visit_index))
                dataframe_index += 1
            subject_index += 1

        return index_translation

    def _get_neighboring_times(
        self,
        window_size,
        index,
    ) -> torch.Tensor:
        """
        Gets neighboring vecs that are in the dataset,
        returning both what the results are
        and a masking vector to zero out
        gradients if they're unusable.

        :param window_size: how far in each direction to look
        :param index : initial index to start from.
        :return: neighboring_vecs.shape=[2*window_size, n_combined_indicators],
            mask.shape = [2*window_size, 1]
        """

        neighboring_vecs = torch.zeros(
            2 * window_size,
            self.n_combined_indicators,
            dtype=torch.float
        )
        mask = torch.zeros(2*window_size, 1, dtype=torch.float)

        subject_index, visit_index = self.index_translation[index]
        time_segment = (
            self.dataframe['visits_data']
            [subject_index]
            [visit_index]
            [IDX_TIME_SEGMENT_NUMBER].as_py()
        )

        # Obtain values for neighboring indices.

        # Start from index - 1 and iterate down to index - window_size or
        # 0 (inclusive), whichever is larger. Only keep things with time
        # segments within the specified window size. Break if we end up
        # at another subject.

        for i, index_i in enumerate(
            range(index - 1, max(-1, index - window_size - 1), -1)
        ):
            sub_idx_back, vis_idx_back = self.index_translation[index_i]
            if sub_idx_back != subject_index:
                break
            neighboring_vecs[window_size - i - 1, :] = (
                self._construct_visit_vec(
                    index_i
                )
            )
            neighbor_time_segment = (
                self.dataframe['visits_data']
                [sub_idx_back]
                [vis_idx_back]
                [IDX_TIME_SEGMENT_NUMBER].as_py()
            )
            if abs(neighbor_time_segment - time_segment) <= window_size:
                mask[window_size - i - 1] = 1.

        # Start from index + 1 and iterate up to index + window_size or
        # len(dataframe) (inclusive), whichever is larger. Only keep things
        # with time segments within the specified window size. Break if
        # we end up at another subject.

        for i, index_i in enumerate(
            range(
                index + 1, min(
                    len(self.index_translation), index + window_size + 1
                )
            )
        ):
            sub_idx_for, vis_idx_for = self.index_translation[index_i]
            if sub_idx_for != subject_index:
                break
            neighboring_vecs[window_size + i, :] = (
                self._construct_visit_vec(
                    index_i
                )
            )
            neighbor_time_segment = (
                self.dataframe['visits_data']
                [sub_idx_for]
                [vis_idx_for]
                [IDX_TIME_SEGMENT_NUMBER].as_py()
            )
            if abs(neighbor_time_segment - time_segment) <= window_size:
                mask[window_size + i] = 1.

        return neighboring_vecs, mask

    def _construct_visit_vec(
        self,
        index
    ) -> torch.Tensor:
        """
        Constructs the vector corresponding to
        indices that are populated are chosen by
        self.medical_code_to_index

        :param index: index for with visit to select
        :return: medical_codes.shape = [n_combined_indicators]
        """
        subject_index, visit_index = self.index_translation[index]

        visits_data = (
            self.dataframe['visits_data']
            [subject_index]
            [visit_index]
        )

        medical_vec = torch.zeros(
            self.n_combined_indicators,
            dtype=torch.float
        )

        medical_vec = _translate_to_vecs(
            medical_vec,
            visits_data,
            self.code_indices,
            self.medical_code_to_index
        )

        return medical_vec

    def _construct_negative_samples(
        self,
        p: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Constructing negative samples - effectively sampling from codes,
        up to a max occupancy assuming a uniform distribution.
        Would be nice to sample according to the true
        one point frequency distribution.

        :param p: the one point distribution. Uniform if
            not specified.
            p.shape = self.n_combined_indicators.
        :return: neg_samples.shape = [n_combined_indicators]
        """

        if p is None:
            p = self.max_occupancy * (
                torch.ones(
                    self.n_combined_indicators)
                / self.n_combined_indicators
            )

        negative_sample = torch.bernoulli(p)

        indices = torch.where(negative_sample)[0]

        if len(indices) > self.max_occupancy:
            selected_indices = torch.randperm(self.max_occupancy)
            indices = indices[selected_indices]
            negative_sample = torch.zeros_like(negative_sample)
            for index in indices:
                negative_sample[index] = 1.

        return negative_sample

    def _get_demographics(
        self,
        use_age: bool,
        use_sex: bool,
        index: int
    ) -> typing.Optional[torch.Tensor]:
        """
        Get the age and sex, per user boolean specifications.

        :param use_age: bool, should we use age or not
        :param use_sex: bool, should we use sex or not.
            assumes two sexes
        :param index: index of the datapoint
        :return: demographic_vec.shape=[1] or [2]
        """

        subject_index, visit_index = self.index_translation[index]

        # Age in years, approximated for leap years.
        # TODO : age calculation needs to be a part of
        # preprocessing in PFL. The below lines
        # will be changed to reflect those changes.
        if use_age:
            current_age = 0.

        if use_sex:
            if self.dataframe['sex'][subject_index].as_py().upper()[0] == "M":
                sex = 0.
            else:
                sex = 1.

        if use_age and use_sex:
            return torch.Tensor([current_age, sex])
        elif use_age and not use_sex:
            return torch.Tensor([current_age])
        elif use_sex and not use_age:
            return torch.Tensor([sex])
        else:
            return torch.Tensor([0.])

    def _get_cooccurrence_pairs(
        self,
        vector: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:
        """
        Gets the pairs of co-occurring codes, up to
        the maximum allowed occupancy. Effectively, this
        is a dense representation of a sparse adjacency matrix.
        This will enable batched computation of code loss.

        :param vector: code vector representation of a
            medical visit.
            vector.shape = self.n_combined_indicators.
        :return: pair_list (or None),
            pair_list.shape = Tensor([number of pairs, 2]).
        """
        if torch.sum(vector) > 1:
            pair_list = list(
                torch.combinations(
                    torch.where(vector)[0],
                    2
                )
            )

            if len(pair_list) > (self.max_occupancy**2-self.max_occupancy)/2.:
                pair_list = random.sample(
                    pair_list,
                    int((self.max_occupancy**2 - self.max_occupancy)/2)
                )
            return torch.stack(pair_list).to(dtype=torch.long)
        return None

    def __getitem__(self, index):
        X = dict()

        X['med_vec'] = self._construct_visit_vec(index)

        X['pos_cooc_pairs'] = self._get_cooccurrence_pairs(X['med_vec'])

        X['negative_samples'] = self._construct_negative_samples()

        X['neg_cooc_pairs'] = self._get_cooccurrence_pairs(
            X['negative_samples']
        )

        X['demographics'] = self._get_demographics(
            self.use_age, self.use_sex, index)

        X['neighbor_vecs'], X['mask'] = self._get_neighboring_times(
            self.window_size, index)

        return X

import torch
from tqdm import tqdm
import typing


def code_loss_per_example(
    weights: torch.nn.parameter.Parameter,
    med_vector: torch.Tensor,
    eps: float = 1e-15
) -> torch.Tensor:
    """
    NOTE: DEPRECATED IN MOST RECENT IMPLEMENTATION OF
    CODE LOSS CALCULATION. LEFT IN FOR POSTERITY, CAN
    DELETE IF NECESSARY.

    Computes the code loss for one example of
    a medical vector.

    :param weights: the weight of the embedder
        layer
    :param med_vector: the vector for which we
        are computing the code loss
    :param eps: a float to keep from taking the log of 0.
    :return: loss, where tensor.shape = [1]
    """

    # weights.shape = [embedding dim, n_codes]

    # weights[:, med_vector.bool()].shape = [embedding dim, occupied]

    numer = torch.exp(
        weights[:, med_vector.bool()].T@weights[:, med_vector.bool()]
    )

    # numer.shape = [occupied, occupied]

    # Zeroing out the diagonal.

    numer = numer - torch.diag(torch.diag(numer))

    denom = torch.sum(
        torch.exp(weights.T@weights[:, med_vector.bool()]), dim=0
    )
    # denom.shape = [occupied]
    return torch.sum(torch.log(numer/denom+eps))


class Med2Vec(torch.nn.Module):
    """
    Implementing a Med2Vec module with feedforward net with variable activation
    """
    def __init__(
        self,
        n_codes: int,
        embedding_dim: int,
        latent_dims: typing.List[int],
        window_size: int,
        use_demographics: bool = False,
        demographic_dim: int = 0,
        code_activation: torch.nn = torch.nn.ReLU(),
        visit_activation: torch.nn = torch.nn.ReLU(),
    ):
        """
        Initializes the med2vec neural network.

        :param n_codes: int, number corresponding
            to the number of possible codes of the
            input to med2vec
        :param embedding_dim: int, dimension of the
            embedding space of med2vec
        :param latent_dims: List[int], dimension of the
            representation from code to visit representation
        :param window_size: int, neighboring windows to look in
        :param use_demographics: bool, do we use demographisc or not
        :param demographics dim: int, size of demographics vector
        :param code_activation: activation function for embedding
        :param visit_activation: activation function for visit
            representation
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.use_demographics = use_demographics
        self.n_codes = n_codes

        self.embedder = torch.nn.Linear(self.n_codes, embedding_dim)
        self.embedder.weight.data.clamp_(0)

        self.code_activation = code_activation

        if self.use_demographics:
            full_latent_dims = [embedding_dim + demographic_dim] + latent_dims
        else:
            full_latent_dims = [embedding_dim] + latent_dims

        layers = []

        self.window_size = window_size

        for i in range(1, len(full_latent_dims)):
            layers.append(
                torch.nn.Linear(
                    full_latent_dims[i-1], full_latent_dims[i]
                )
            )
            layers.append(visit_activation)

        self.MLP = torch.nn.Sequential(*layers)

        self.visit_to_code = torch.nn.Linear(
            full_latent_dims[-1], self.n_codes)
        self.code_to_prediction = torch.nn.Softmax(dim=-1)

        # Move the appropriate elements to the correct device.
        self.embedder.to(self.device)
        self.code_activation.to(self.device)
        self.MLP.to(self.device)
        self.visit_to_code.to(self.device)
        self.code_to_prediction.to(self.device)

    @staticmethod
    def _cross_ent_loss(
        targets: torch.Tensor,
        predictions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates cross entropy loss manually.
        Implementing using torch's inbuilt would
        work but I'm running into issues with
        model instability.

        :param targets: the tensor containing the true codes being targeted
        :param predictions: tensor containing predicted probits
        :param mask: indicator variable telling us which predictions can be
            considered
        :return: cross_ent_loss.shape = [1]
        """

        # Targets.shape = [batch_size, 2*window size, n_combined indicators].
        # Predictions.shape = [batch_size, n_combined_indicators].

        # We start by matching the sizes of predictions and targets.

        predictions = predictions.unsqueeze(1)
        predictions = predictions.tile([1, targets.shape[1], 1])

        # Compute loss on a per site basis, and sum over the
        # n_combined_indicators so that we can apply the mask.

        loss = torch.sum(
            torch.nn.BCELoss(reduction='none')(predictions, targets),
            dim=-1,
            keepdim=True
        )

        # Loss.shape = [batch_size, 2 * window_size, 1].

        return torch.sum(loss * mask) / predictions.shape[0]

    def _code_loss(
        self,
        x: torch.Tensor,
        neg_samples: typing.Optional[torch.Tensor] = None,
        eps: float = 1e-15
    ) -> torch.Tensor:
        """
        Computes code loss by ensuring the
        coocurrence probability (or log thereof) is maximized.
        We calculate this by identifying the indices for which
        a diagnostic code appears for each vector, and then
        slicing the code embedding matrix and obtaining the
        dot products accordingly.

        Target equation:

        log(exp(W[:,i]^{T} W[:,j])/(sum_{k} exp(W^{T} W[:,j]))

        :param x: real pairs of cooccurrence
        :param neg_samples: negative pairs of cooccurrence
        :return: code_loss.shape = [1]
        """

        loss = 0.

        weights = self.embedder.weight

        pos_numer = torch.diag(
            torch.exp(
                weights[:, x[:, 0]].T@weights[:, x[:, 1]]
            )
        )
        pos_denom = torch.sum(torch.exp(weights.T@weights[:, x[:, 0]]), dim=0)

        loss -= torch.sum(torch.log(pos_numer/pos_denom + eps))

        neg_numer = torch.diag(
            torch.exp(
                weights[:, neg_samples[:, 0]].T@weights[:, neg_samples[:, 1]]
            )
        )
        neg_denom = torch.sum(
            torch.exp(
                weights.T@weights[:, neg_samples[:, 0]]
            ),
            dim=0
        )

        loss += torch.sum(torch.log(neg_numer/neg_denom + eps))

        return loss/((self.n_codes**2 - self.n_codes)/2)

    def forward(
        self,
        x: dict
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass of the model. Returns the
        prediction, code embedding, and visit embedding.

        parameters:
        :param x: dict from dataloader with a 'med_vec' and
            'demographics' key
        :return:
            yhat.shape = [batch_size, n_combined_indicators],
            code_embedding.shape = [batch_size, code_dim],
            visit_embedding.shape = [batch_size, latent_dims[-1]]
        """

        if self.use_demographics:
            input_code, demographics = x['med_vec'], x['demographics']
        else:
            input_code = x['med_vec']

        code_embedding = self.code_activation(self.embedder(input_code))

        if self.use_demographics:
            x = torch.concat((code_embedding, demographics), dim=1)
        else:
            x = code_embedding

        visit_embedding = self.MLP(x)

        yhat = torch.nn.Softmax(dim=-1)(self.visit_to_code(visit_embedding))

        return yhat, code_embedding, visit_embedding

    def _train(
        self,
        optim: torch.optim,
        x: dict,
        y: dict
    ) -> float:
        """
        One training step of the med2vec training algorithm.
        Done over one minibatch
        Written to ensure code embedding weights remain positive

        :param optim: torch optimizer. Typically SGD.
        :param x: dict with 'med_vec', 'demographics' (optional), and
            'negative_samples' keys
        :param y: dict with 'neighbor_vecs' and 'mask' key
        :return: loss.item(). It is one float
        """

        optim.zero_grad()

        yhat, code_embedding, visit_embedding = self.forward(x)

        if self.use_demographics:
            yhat, _, _ = self.forward(x)
        else:
            yhat, _, _ = self.forward(x)

        visit_loss = self._cross_ent_loss(y['neighbor_vecs'], yhat, y['mask'])
        code_loss = self._code_loss(x['pos_cooc_pairs'], x['neg_cooc_pairs'])

        loss = visit_loss + code_loss
        loss.backward()

        optim.step()

        self.embedder.weight.data.clamp_(0)

        return loss.item()

    def val(
        self,
        x: dict,
        y: dict
    ) -> float:
        """
        One validation step of med2vec embedding.

        parameters:
        :param x: dict with keys 'med_vec' and 'demographics' (opt)
        :param y: dict with 'mask' and 'neighbor_vecs'
        :return: loss.item()
        """
        with torch.no_grad():
            yhat, _, _ = self.forward(x)
            loss = self._cross_ent_loss(y['neighbor_vecs'], yhat, y['mask'])
            loss += self._code_loss(x['pos_cooc_pairs'], x['neg_cooc_pairs'])
        return loss.item()

    def fit(
        self,
        epochs: int,
        dg_train: torch.utils.data.DataLoader,
        dg_val: typing.Optional[torch.utils.data.DataLoader],
        optim: torch.optim
    ) -> (typing.List[float], typing.List[float]):
        """
        Runs the training loop for specified number of epochs.

        :param epochs: int, how many epochs to run training for?
        :param dg_train: datagenerator for training only.
        :param dg_val: datagenerator, optional, for validation
        :param optim: optimizer specified by user. SGD is a
            good place to start if unsure
        :return: training_loss_tracker, val_loss_tracker
        """

        training_loss_tracker = []
        val_loss_tracker = []

        for epoch in tqdm(range(epochs)):
            training_loss = 0
            val_loss = 0

            for batch in dg_train:
                # Split batch into X and Y dicts.
                X_keys = [
                    'med_vec',
                    'pos_cooc_pairs',
                    'negative_samples',
                    'neg_cooc_pairs',
                    'demographics'
                ]
                y_keys = ['neighbor_vecs', 'mask']

                X = {k: batch[k].to(self.device) for k in X_keys}
                y = {k: batch[k].to(self.device) for k in y_keys}

                training_loss += self._train(optim, X, y)
            training_loss_tracker.append(training_loss)

            if dg_val is not None:
                for batch in dg_val:
                    X_keys = [
                        'med_vec',
                        'pos_cooc_pairs',
                        'negative_samples',
                        'neg_cooc_pairs',
                        'demographics'
                    ]
                    y_keys = ['neighbor_vecs', 'mask']
                    X = {k: batch[k].to(self.device) for k in X_keys}
                    y = {k: batch[k].to(self.device) for k in y_keys}
                    val_loss += self.val(X, y)
                val_loss_tracker.append(val_loss)

        return training_loss_tracker, val_loss_tracker

import torch
from sklearn.mixture import GaussianMixture
import numpy as np
import typing


class VariationalDeepEmbedding(torch.nn.Module):
    """
    Implementing Variational Deep embedding.
    This differs from traditional variational
    autoencoders by virtue of being a gaussian
    mixture model as the prior. In principle,
    one could imagine that this would approach
    a VAE if there was only one gaussian
    in the GMM.
    See:
    VAE: https://arxiv.org/abs/1312.6114.
    VaDE: https://arxiv.org/abs/1611.05148.
    InfoVAE:https://arxiv.org/abs/1706.02262
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        gaussian_centers: int,
        MMD: bool = False,
        targets: typing.List = []
    ):
        """
        Initializing Variational Deep Embedding.
        We require the number of gaussian centers
        initially, as well as an encoder and a decoder.
        The latent dimensions must be specified and
        congruent with what appears in the encoder and
        decoder. Gaussian centers for the number of gaussians
        should also be specified.

        :param encoder: torch.nn.Module. A neural network
            that performs the encoding to latent space.
        :param decoder: torch.nn.Module. A neural network
            that decodes from latent space.
        :param latent_dim: int. Dimensionality of the latent space.
        :param gaussian_centers: int. How many gaussian centers
            comprise the mixture of gaussians prior.
        :param MMD: Specifies if we want to run in MMD mode.
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.gaussian_centers = gaussian_centers
        self.mu_c = torch.nn.Parameter(
            torch.zeros(gaussian_centers, latent_dim), requires_grad=True
        )
        self.log_var_c = torch.nn.Parameter(
            torch.zeros(gaussian_centers, latent_dim), requires_grad=True
        )
        self.log_pi_c = torch.nn.Parameter(
            torch.zeros(gaussian_centers), requires_grad=True
        )
        self.MMD = MMD

        self.predictors = {}

        for target in self.targets:
            self.predictors[target] = (
                torch.nn.Linear(self.latent_dim, 1).to(self.device)
            )

    def _fit_gmms(
        self,
        z: torch.Tensor
    ) -> GaussianMixture:
        """
        Fits the GMMs given a tensor of latent embeddings using
        sklearn's Gaussian Mixture model.
        :param z: torch.Tensor.shape = [examples, latent_dim].
            Used for fitting the GMM
        :return: A fit gaussian mixture model, assuming diagonal
            covariance.
        """
        test_z = z.detach().numpy()
        gmm = GaussianMixture(
            n_components=self.gaussian_centers, covariance_type='diag'
        )
        gmm.fit(test_z)

        if self.gaussian_centers > 1:
            self.mu_c.data = torch.from_numpy(gmm.means_)
            self.log_var_c.data = torch.from_numpy(np.log(gmm.covariances_))
            # self.pi_c.data = torch.from_numpy(gmm.weights_)

        return gmm

    @staticmethod
    def reparametrization(
        z_mu: torch.Tensor,
        z_log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparametrization trick, as per stochastic gradient variational
        bayes. Enables differentiation through the encoder.
        :param z_mu: torch.Tensor.shape = [batch_size, latent_dim].
            Means of the encoding.
        :param z_log_var: torch.Tensor.shape = [batch_size, latent_dim].
            Log variances of the encoded coordinate

        :return: z, torch.Tensor.shape = [batch_size, latent_dim]. Each
            entry will be a sample from a gaussian distribution with
            mean given by z_mu and variance given by exp(z_log_var).
        """
        z = z_mu + torch.exp(z_log_var/2.)*torch.randn_like(z_mu)
        return z

    def forward(
        self,
        X: torch.Tensor,
        mask: typing.Optional[torch.Tensor] = None
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Implements the forward pass of the variational model.

        :param X: torch.Tensor.shape = [batch_size, input_dim].
            Object we are trying to embed in a latent space.
        :param mask: torch.Tensor.shape = [batch_size, input_dim].
            Masking for not present visits in the dataset.
        :return: z, torch.Tensor.shape = [batch_size, latent_dim].
            Latent embeddings post reparametrization.
        :return: z_mu, torch.Tensor.shape = [batch_size, latent_dim].
            Mean of latent embeddings.
        :return: z_log_var, torch.Tensor.shape = [batch_size, latent_dim].
            Log variance of latent embeddings.
        :return: pred, torch.Tensor.shape = X.shape.
            Reconstructed input.
        """
        if mask is not None:
            result = self.encoder(X, mask)
        else:
            result = self.encoder(X, mask)
        z_mu = result['z_mu']
        z_log_var = result['z_log_var']

        z = self.reparametrization(z_mu, z_log_var)
        if self.MMD:
            pred = self.decoder(z_mu)
        else:
            pred = self.decoder(z)

        outputs = {}
        for idx, target in enumerate(self.targets):
            outputs[target] = self.predictors[target](z_mu)

        outputs['z'] = z
        outputs['z_mu'] = z_mu
        outputs['z_log_var'] = z_log_var
        outputs['pred'] = pred['Xhat']

        return outputs

    def total_loss(
        self,
        z: torch.Tensor,
        z_mu: torch.Tensor,
        z_log_var: torch.Tensor,
        pred: torch.Tensor,
        true: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        This method computes the total loss. This is the reconstruction
        loss plus the prior loss.

        :param z: torch.Tensor, shape = [batch_size, latent_dim].
        :param z_mu: torch.Tensor, shape = [batch_size, latent_dim].
        :param z_log_var: torch.Tensor, shape = [batch_size, latent_dim].
        :param pred: torch.Tensor, shape = [batch_size, input_dim].
        :param true: torch.Tensor, shape = [batch_size, input_dim].
        :param alpha: float, weight of reconstruction loss relative
            to prior loss.

        :return: loss, torch.Tensor, shape=1.
        """
        if self.MMD:
            return (
                alpha * self.recon_loss(true, pred)
                + self.mmd_loss(z_mu)
            )
        else:
            return (
                alpha*self.recon_loss(true, pred)
                + self.elbo_loss(z_mu, z_log_var)
            )

    @staticmethod
    def recon_loss(
        true: torch.Tensor,
        pred: torch.Tensor
    ) -> torch.Tensor:
        """
        A method for computing reconstruction loss using
        in-built BCE. In practice, this is not useful
        if we have to consider missing values.
        :param true: The target for reconstruction.
            true.shape = [batch_size, input_dim].
        :param pred: The predicted reconstruction.
            pred.shape = [batch_size, input_dim].

        :return: loss, torch.Tensor.
        """
        loss = torch.nn.BCELoss(reduction='mean')(pred, true)
        return loss

    def mmd_loss(
        self,
        z: torch.Tensor,
        num_samples: int = 200
    ) -> torch.Tensor:
        """
        Computes loss based on maximum-mean discrepancy,
        in order to optimize infoVAE objective function.

        The objective is typically written down as
        E[k(z_true, z_true)] + E[k(z_encoder, z_encoder)]
        - 2 E[k(z_true, z_encoder)]

        where k(x, y) is a positive definite kernel, typically
        taken to be a reproducing kernel hilbert space, such
        as the negative quadratic exponential.

        :param z: Outputs if the encoder. z.shape = [batch_size, latent_dim]
        :param num_samples: Number of samples to generate from the true prior
            distribution

        :return: mmd_loss, torch.Tensor, shape=1
        """
        if self.gaussian_centers == 1:
            z_prior = torch.randn(num_samples, z.shape[-1]).to(self.device)
        else:
            mix = torch.distributions.Categorical(torch.exp(self.log_pi_c))
            mean_and_vars = torch.distributions.Independent(
                torch.distributions.Normal(
                    self.mu_c, torch.exp(self.log_var_c/2.)
                ),
                1
            )
            gmm = torch.distributions.MixtureSameFamily(mix, mean_and_vars)
            z_prior = gmm.sample(torch.tensor([num_samples])).to(self.device)

        def calc_kernel(
            x: torch.Tensor,
            y: torch.Tensor
        ):
            """
            Calculate pairwise distances between tensors
            x and y.
            """
            dim = x.shape[1]
            x = x.unsqueeze(0)
            y = y.unsqueeze(1)

            return torch.exp(-torch.mean((x-y)**2, -1)/dim)

        Exx = torch.mean(calc_kernel(z_prior, z_prior))
        Eyy = torch.mean(calc_kernel(z, z))
        Exy = torch.mean(calc_kernel(z, z_prior))

        return Exx + Eyy - 2 * Exy

    def elbo_loss(
        self,
        z_mu: torch.Tensor,
        z_log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Evidence Lower Bound Loss, for a Variational Autoencoder. If training
        in VAE mode, use recon_loss + elbo_loss, rather than total loss.

        loss = -1/2 (1 + log(var) - mu^2 - var)

        :param z_mu: mean latent embedding. shape=[batch_size, latent_dim].
        :param z_log_var: log variance of embedding.
            shape = [batch_size, latent_dim].

        :return: loss, torch.Tensor, shape=1
        """
        loss = -0.5 * torch.mean(
            1 + z_log_var - z_mu**2 - torch.exp(z_log_var)
        )
        return loss

    def prior_loss(
        self,
        z: torch.Tensor,
        z_mu: torch.Tensor,
        z_log_var: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Evidence Lower Bound Loss for variational deep embedding.
        Computes loss against a mixture of gaussian models, simultaneously
        updating the gaussian mixture model and the encoder.

        loss = -1/2 * (gamma_c * (log(var_c)
            + log(var)/log(var_c) + (mu - mu_c)**2 / log(var_c)))
            +gamma_c * log(pi_c/gamma_c) + 1/2 * (1+log(var)).

        :param z: The latent embeddings for the encoder.
            shape = [batch_size, latent_dim].
        :param z_mu: The mean embeddings thereof. shape = z.shape
        :param z_log_var: The log variance embeddings thereof. shape = z.shape.
        :param eps: float, infinitesimal number to prevent divide by 0s.

        :return: loss, torch.Tensor, shape=1.
        """
        mu_c = self.mu_c
        log_var_c = self.log_var_c
        log_pi_c = self.log_pi_c

        loss_qz_giv_x = -1./2. * torch.mean(torch.sum(1 + z_log_var, dim=-1))
        loss_pz_giv_c = +1./2. * (
            log_var_c
            + torch.exp(- log_var_c + z_log_var.unsqueeze(1))
            + (z_mu.unsqueeze(1) - mu_c)**2 / (torch.exp(log_var_c) + eps)
        )

        gamma_c = self.cluster_proba(z)

        pi_c = torch.exp(log_pi_c)/torch.sum(torch.exp(log_pi_c))

        loss_qc_giv_x = -torch.mean(
            torch.sum(gamma_c * torch.log(pi_c/(gamma_c+eps)), dim=-1)
        )

        loss_pz_giv_c = torch.mean(
            torch.sum(gamma_c * torch.sum(loss_pz_giv_c, dim=-1), dim=-1)
        )

        return loss_qz_giv_x + loss_pz_giv_c + loss_qc_giv_x

    def cluster_proba(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the probability of belonging
        to each gaussian center in the mixture of gaussians prior.

        :param z: Latent embeddings, shape=[batch_size, latent_dim].

        :return: proba, probabilities of belonging to each gaussian center.
            shape = [batch_size, gaussian_centers].
        """
        mu_c = self.mu_c
        log_var_c = self.log_var_c
        log_pi_c = self.log_pi_c

        pi_c = torch.exp(log_pi_c)/torch.sum(torch.exp(log_pi_c))

        unsqueezed_z = z.unsqueeze(1)
        unsqueezed_means = mu_c.unsqueeze(0)
        unsqueezed_log_covariances = log_var_c.unsqueeze(0)

        mean_deviation = unsqueezed_z - unsqueezed_means
        exponent = torch.sum(
            mean_deviation * 1./torch.exp(
                unsqueezed_log_covariances
            ) * mean_deviation,
            dim=-1
        )

        coefficients = (2 * torch.pi)**(-self.latent_dim/2) * torch.exp(
            -torch.sum(unsqueezed_log_covariances, dim=-1)
        )

        raw_probs = coefficients * torch.exp(-(exponent/2)) + 1e-10
        probs = raw_probs * pi_c
        gamma_c = probs / torch.sum(probs, dim=-1).unsqueeze(1)
        return gamma_c


class VaDER(VariationalDeepEmbedding):
    """
    Implements a VaDER model, based on a paper that observed
    it performed well because of its learned imputation kernel
    and its ability to construct a potentially valuable embedding
    space.
    See VADER reference: https://doi.org/10.1093/gigascience/giz134.
    """
    def __init__(
        self,
        n_steps: int,
        hidden_layer: int,
        latent_dim: int,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        gaussian_centers: int
    ):
        """
        Initializes the vader, tracking all the important parameters of the
        model.

        :param n_steps: int, Length of time series.
        :param hidden_layer: int, Dimensionality of hidden and cell state.
        :param latent_dim: int, Dimensionality of latent space.
        :param encoder: Encoder object for Vader.
        :param decoder: Decoder object for Vader.
        :param gaussian_centers: int, number of gaussian centers.
        """
        super().__init__(encoder, decoder, latent_dim, gaussian_centers)
        self.n_steps = n_steps
        self.hidden_layer = hidden_layer

    @staticmethod
    def recon_loss(
        true: torch.Tensor,
        pred: torch.Tensor,
        missing_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementing MSE loss for reconstruction, as that seems
        somewhat appropriate for this limit. Can be rewritten to
        be something else, though.
        All inputs are shaped: [batch_size, n_steps, input_size].

        :param true: The true time series.
        :param pred: The predicted time series.
        :param missing_mask: Binary tensor of what
            entries will be missing.

        :return: loss, shape=1
        """
        mse_raw = (true-pred)**2
        return torch.mean((1-missing_mask)*mse_raw)

    def total_loss(
        self,
        z: torch.Tensor,
        z_mu: torch.Tensor,
        z_log_var: torch.Tensor,
        pred: torch.Tensor,
        true: torch.Tensor,
        missing_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes total loss of the model.

        :param z: Latent embeddings, shape=[batch_size, latent_dim].
        :param z_mu: Mean latent embeddings.
        :param z_log_var: Log variance of latent embeddings.
        :param pred: Reconstructed object.
            Shape = [batch_size, n_steps, input_size].
        :param true: true time series.
        :param missing_mask: Binary tensor of missing entries.
        :return: Loss, shape=1.
        """
        return (
            100*self.recon_loss(
                true, pred, missing_mask
            )
            + self.prior_loss(z, z_mu, z_log_var)
        )

    def forward(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, ...]:
        """
        Implements forward pass of the model, given the input
        time series and the missing mask.

        :param X: Input time series. Shape = [batch_size, n_steps, input_size].
        :param missing_mask: Boolean tensor of missing entries.

        :return: z, z_mu, z_log_var, pred
            z: Latent embeddings of model.
                z.shape = [batch_size, latent_dim].
            z_mu: Mean embeddings.
            z_log_var: Log variance of embeddings.
            pred: Reconstruction prediction.
                pred.shape = [batch_size, n_steps, input_dim].
        """
        z_mu, z_log_var = self.encoder(X, missing_mask)
        z = self.reparametrization(z_mu, z_log_var)
        pred = self.decoder(z)
        return z, z_mu, z_log_var, pred