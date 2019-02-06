# importing dependencies 
import numpy as np 
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
import scipy.optimize as sco
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt 
from scipy.stats import norm

def dist_moments(x):
    return np.mean(x), np.std(x), skew(x), kurtosis(x)

def annretvol_asset(asset_return, year_days):
    mean_returns, returns_std, _, _, = dist_moments(asset_return)
    year_return = mean_returns * year_days
    year_volatility = returns_std * np.sqrt(year_days)

    annualized_return = year_return * 100
    annualized_volatility = year_volatility * 100

    return annualized_return, annualized_volatility

def annretvol_port(asset_returns, asset_weights, year_days):
    mean_daily_returns = asset_returns.mean()
    returns_cov = asset_returns.cov()
    
    portfolio_return = np.sum(mean_daily_returns * asset_weights) * year_days
    portfolio_volatility = np.sqrt(np.dot(asset_weights.T, np.dot(returns_cov, asset_weights))) * np.sqrt(year_days)

    portfolio_return = portfolio_return * 100
    portfolio_volatility = portfolio_volatility * 100 

    return portfolio_return, portfolio_volatility

def annvol_port(asset_weights, asset_returns, year_days):
    return annretvol_port(asset_returns, asset_weights, year_days)[1]

def annretvol_port_rand(asset_returns, year_days, n_simulations=100, seed=None):
    # seed numpy random
    np.random.seed(seed)

    # get total number of assets
    n_assets = asset_returns.shape[1]

    # init array to hold results
    random_weight_results = np.zeros((4, n_simulations))
    weights_assigned = []

    # loop for the number of simulations specified 
    for i in range(n_simulations):
        # generate random weights
        random_weights = np.random.random(n_assets)
        # calibrate to be equal to 1
        random_weights /= np.sum(random_weights)
    
        # calculate portfolio annualized expected return and volatility using random weights 
        portfolio_returns, portfolio_volatility = annretvol_port(asset_returns=asset_returns,
                                                                asset_weights=random_weights,
                                                                year_days=year_days)
    
        # set results
        # expected returns result 
        random_weight_results[0,i] = portfolio_returns 
        # expected volatility result
        random_weight_results[1,i] = portfolio_volatility 
        # sharpe ratio result risk free rate set to 0
        random_weight_results[2,i] = portfolio_returns / portfolio_volatility 
        # Var 99%
        random_weight_results[3,i] = var_cov_var(random_weights, asset_returns, year_days)

        # weights assigned
        weights_assigned.append(random_weights)

    # convert results to dataframe
    results_df = pd.DataFrame(random_weight_results.T, 
                            columns=["expected_return", "expected_volatility", "sharpe_ratio", "var_99"])
    
    # add weights to the dataframe
    results_df["weights"] = weights_assigned

    # return results 
    return results_df

def neg_sharperatio(asset_weights, asset_returns, year_days, risk_free_rate=0):
    returns, volatility = annretvol_port(asset_returns,asset_weights,year_days)
    return -(returns-risk_free_rate) / volatility

def max_sharperatio_port(asset_returns, year_days ,risk_free_rate=0):
    
    # get total number of assets
    n_assets = asset_returns.shape[1]

    # arguments for minimize function
    args = (asset_returns, year_days, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # constraints for function
    bounds = tuple((0,1) for asset in range(n_assets)) # bounds between 0 and 1

    # optimize sharpe ratio (finding maximum)
    optimization =sco.minimize(neg_sharperatio, 
                               n_assets*[1./n_assets,], 
                               args=args, 
                               method='SLSQP', 
                               bounds=bounds, 
                               constraints=constraints)

    # return results
    weights = optimization["x"] 
    returns, volatility = annretvol_port(asset_returns, weights, year_days)     
    return returns, volatility, ((returns-risk_free_rate) / volatility), weights

def min_volatility_port(asset_returns, year_days):
    # get total number of assets
    n_assets = asset_returns.shape[1]
    
    # arguments for minimize function
    args = (asset_returns, year_days)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # constraints for function
    bounds = tuple( (0,1) for asset in range(n_assets)) # bounds between 0 and 1

    # optimize volatility (finding minimum)
    optimization = sco.minimize(annvol_port, 
                                n_assets*[1./n_assets,], 
                                args=args,
                                method='SLSQP', 
                                bounds=bounds, 
                                constraints=constraints)

    # return results
    weights = optimization["x"] 
    returns, volatility = annretvol_port(asset_returns, weights, year_days)     
    return returns, volatility, weights

def min_var_port(asset_returns, year_days, c=2.33):
    # get total number of assets
    n_assets = asset_returns.shape[1]
    
    # arguments for minimize function
    args = (asset_returns, year_days, c)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # constraints for function
    bounds = tuple( (0,1) for asset in range(n_assets)) # bounds between 0 and 1

    # optimize var (finding minimum)
    optimization = sco.minimize(var_cov_var, 
                                n_assets*[1./n_assets,], 
                                args=args,
                                method='SLSQP', 
                                bounds=bounds, 
                                constraints=constraints)

    # return results
    weights = optimization["x"] 
    returns, volatility = annretvol_port(asset_returns, weights, year_days)   
    var = var_cov_var(weights, asset_returns, year_days, c)
    return returns, volatility, var, weights

def var_cov_var(asset_weights, asset_returns, year_days, c=2.33):
    volatility = annvol_port(asset_weights, asset_returns, year_days) 
    var = c * volatility
    return var

def garch_vol(returns, verbose=True):
    garch = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
    result = garch.fit(update_freq=5)

    if verbose:
        print(result.summary())

    cond_vol = result.conditional_volatility.iloc[-1]
    return cond_vol 

def beta_test_ols(x, y, x_label, y_label):
    x1 = sm.add_constant(x)
    model = sm.OLS(y, x1)
    results = model.fit()

    beta = round(results.params[1], 4)
    alpha = round(results.params[0], 4)
    
    print("{0} against {1} Beta-test".format(y_label, x_label) +
      "\nSummary (OLS): {0}".format(results.summary()) +
      "\n\nBeta: {0}".format(beta) + 
      "\nWith 97.5% confidence lies between {0} and {1}".format(round(results.conf_int(alpha=0.05, cols=None)[0][1],4),
                                                                round(results.conf_int(alpha=0.05, cols=None)[1][1],4)) + 
      "\nP-value is: {0}; P-Value < 0.05: {1}".format(results.pvalues[1], (results.pvalues[1] < 0.05)))
    
    x2 = np.linspace(x.min(), x.max(), 100)
    pred_y = x2 * beta + alpha
    plt.figure(figsize=(10,7))
    plt.scatter(x, y, alpha=.3)
    plt.xlabel(x_label + " Daily Return")
    plt.ylabel(y_label + " Daily Return")
    plt.plot(x2, pred_y, "r", alpha = 0.9)
    plt.show()

def binomial_tree(expected_return, expected_volatility, current_price, simulation_time, time_periods, verbose=True):
    delta_t = simulation_time / time_periods
    up_factor = np.exp(expected_volatility/100 * np.sqrt(delta_t))
    down_factor = 1 / up_factor
    up_prob = (np.exp(expected_return/100 * delta_t) - down_factor) / (up_factor - down_factor)
    down_prob = 1 - up_prob
    if verbose:
        print("Current price: {0}".format(current_price) + 
             "\nExpected return: {0}%".format(expected_return) +
             "\nExpected volatility: {0}%".format(expected_volatility) +
             "\n\nSimulation time: {0}".format(simulation_time) +
             "\nTime periods: {0}".format(time_periods) +
             "\nDelt_t: {0}".format(delta_t) +
             "\nUp-factor: {0}".format(up_factor) +
             "\nDown-factor: {0}".format(down_factor) + 
             "\nUp-probability: {0}".format(up_prob) + 
             "\nDown-probability: {0}".format(down_prob))
    
    # columns for data frame 
    columns = ["s0"]
    
    # binomial prices 
    # initialize array for prices
    binomial_prices = np.empty((time_periods + 1 , time_periods + 1))
    binomial_prices[:] = 0
    
    # fill binomial prices
    binomial_prices[0,0] = current_price # set s0 price

    for i in range(time_periods):
        columns.append("s" + str(i+1))
        binomial_prices[0, i + 1] = round(binomial_prices[0, i] * up_factor, 2)
        binomial_prices[1, i + 1] = round(binomial_prices[0, i] * down_factor, 2)
        for j in range(i):
            binomial_prices[j + 2, i + 1] = round(binomial_prices[j + 1, i] * down_factor, 2)
    
    # binomial prices 
    # initialize array for prices
    binomial_probs = np.empty((time_periods + 1 , time_periods + 1))
    binomial_probs[:] = 0
    
    # fill binomial probabilities 
    binomial_probs[0,0] = 1.0 # set s0 prob
    for i in range(time_periods):
        binomial_probs[0, i + 1] = round(binomial_probs[0, i] * up_prob, 4)
        for j in range(i+1):
            prev_prob =  (binomial_probs[j + 1, i] * up_prob)
            binomial_probs[j + 1, i + 1] = round((binomial_probs[j, i] * down_prob) + prev_prob, 4)
    

    # expected values 
    expected_values = np.around(np.sum(binomial_prices * binomial_probs, axis = 0), 2)
    expected_values = np.reshape(expected_values, (-1, 1))
    
    # build dataframes 
    binomial_prices[binomial_prices == 0] = "nan"
    binomial_prices_df = pd.DataFrame(data=binomial_prices, columns=columns)
    binomial_probs[binomial_probs == 0] = 'nan'
    binomial_probs_df = pd.DataFrame(data=binomial_probs, columns=columns)
    expected_values_df = pd.DataFrame(data=expected_values.T, columns=columns)
    
    return binomial_prices_df, binomial_probs_df, expected_values_df
    