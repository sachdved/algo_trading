import yfinance as yf
import datetime
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp

from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

import random

from tqdm import tqdm

def vwap(ticker):
    """
    calculates volume weighted average price of an asset at most recent tick.

    parameters:
        ticker: yfinance.Ticker object
    """
    bid_size, ask_size, ask_price, bid_price = ticker.info['bidSize'], ticker.info['askSize'], ticker.info['bid'], ticker.info['ask']

    return (bid_size * bid_price + ask_size * ask_price)/(bid_size + ask_size)

def odds_calculator(
    spread_cost, 
    max_payout, 
    min_payout = 0, 
    transaction_cost = 0.0067, 
    num_legs = 1
):
    """
    computes probability of landing between two price points given the cost of a spread around those two price points
    and the max_payout and min_payout, considering transaction costs

    Parameters:
        spread_cost: the cost of the spread. Typically thinking about iron condors here
        max_payout: maximum possible payout
        min_payout: minimum payout, potentially 0
        transaction cost = 0.0067, typical on TD Ameritrade
        num_legs: number of legs in options spread. my iron condors have 4 legs
    """
    transaction_cost = num_legs * transaction_cost
    p = - (min_payout - spread_cost - transaction_cost) / ((max_payout - spread_cost - transaction_cost) - (min_payout - spread_cost - transaction_cost))
    return p

def simulate_outcomes(ticker, target_date, distribution, weights):
    """
    simulates the outcomes of a stock price by randomly drawing from the distribution of 2 minute returns between now and a target datetime object.
    currently, distribution is empirical, but advances could be made by implementing a (truncated) levy distro

    parameters:
        ticker: yfinance.Ticker object
        target_date: datetime object identifying the end time
        distribution: range of log returns to draw
        weights: probability of hitting a given bin_edge (the cdf)
    """

    #get time between
    current_date = datetime.datetime.now()
    time_delta = target_date - current_date

    intervals_to_expiry = time_delta.days * (15 + 6*30) + np.min([15 + 6*30, (time_delta.seconds//120)])
    print(intervals_to_expiry)
    log_returns_deviation = np.zeros(100000)

    
    for r in tqdm(range(len(log_returns_deviation))):
        for i in range(intervals_to_expiry):
            index_list = np.where(random.uniform(0,1)>weights)[0]
            if len(index_list)==0:
                index = 0
            else:
                index = np.max(index_list) + 1
            log_returns_deviation[r] += distribution[index]
    
    current_price = vwap(ticker)

    return np.exp(np.log(current_price) + log_returns_deviation)

def laplace_fit(x, b, a):
    """
    PDF of a double-sided laplace distribution.
    This is typically fit via scipy.optimize.curve_fit

    Parameters:
        x: values at which to evaluate pdf
        b: 
    """
    return 1/(2*b) * np.exp(-np.abs(x - a) / b)