###### importing dependencies ############################################################################
import numpy as np 
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
import scipy.optimize as sco
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt 
from scipy.stats import norm
import fintech.constants as const  

###### finance models and functions ######################################################################
def dist_moments(x):
    """ Calculate the four distribution moments. 

        Args:
            x (numpy array): The array used to calculate the distribution moments for. 
        
        Returns:
            numpy array: 1st distribution moment which is the mean of the array.
            numpy array: 2nd distribution moment which is the standard deviation of the array.
            numpy array: 3rd distribution moment which is the skew of the array.
            numpy array: 4th distribution moment which is the kurtosis of the array. 
    """
    return np.mean(x), np.std(x), skew(x), kurtosis(x)

def annretvol_asset(asset_return, year_days):
    """ Annualize the returns and volatility for an asset.

        Args:
            asset_return (numpy array): An array with the returns for an asset. 
            year_days (int): The number of days to be used as a full year.

        Returns:
            float: The annualized returns for an asset.  
            float: The annualized volatility for an asset.  
    """

    # get 1st moment and 2nd moment
    mean_returns, returns_std, _, _, = dist_moments(asset_return)
    
    # annualize returns 
    year_return = mean_returns * year_days
    
    # annaulize volatility 
    year_volatility = returns_std * np.sqrt(year_days)

    # convert to percentages
    annualized_return = year_return * 100
    annualized_volatility = year_volatility * 100
    
    # return results
    return annualized_return, annualized_volatility

def annretvol_port(asset_returns, asset_weights, year_days):
    """ Calculates the annualized portfolio returns and volatility. 

        Args:
            asset_returns (pandas dataframe): Dataframe containing the log returns for each asset. 
            asset_weights (numpy array): An array with the weights set to each asset. 
            year_days (int): The number of days to be used as a full year.

        Returns:
            float: The portfolio expected return as a percentage. 
            float: The portfolio expected volatility as a percentage.  
    """
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
    """ Create different portfolio using random weights. 

        Args: 
            asset_returns (pandas dataframe): Dataframe containing the log returns for each asset. 
            year_days (int): The number of days to be used as a full year.
            n_simulations (int): By default set to 100, the number of simulations. 
            seed (int): A int used to seed the random. 

        Returns:
            pandas dataframe: Dataframe with the following columns ['expected_return', 'expected_volatility'
                                                                    'sharpe_ratio', 'var_99', 'weights']
    """
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
    
    # returns results
    return results_df

def neg_sharperatio(asset_weights, asset_returns, year_days, risk_free_rate=0):
    """ Calculates the negative Sharpe Ratio given a set of inputs. 

        Args:
            asset_weights (numpy array): The weights assigned to each asset.
            asset_returns (pandas dataframe): Dataframe containing the log returns for each asset. 
            year_days (int): The number of days to be used as a full year.
            risk_free_rate (float): The risk free rate by default set to 0. 

        Returns:
            float: The negative Sharpe Ratio for the given inputs. 
    """ 

    # get the returns and volatility for the portfolio 
    returns, volatility = annretvol_port(asset_returns, asset_weights, year_days)
    
    # returns the negative Sharpe Ratio 
    return -((returns-risk_free_rate) / volatility)

def max_sharperatio_port(asset_returns, year_days ,risk_free_rate=0):
    """ Finds the maximum Sharpe Ratio for the portfolio. 

        Args:
            asset_returns (pandas dataframe): Dataframe containing the log returns for each asset. 
            year_days (int): The number of days to be used as a full year.
            risk_free_rate (float): The risk free rate by default set to 0. 

        Returns: 
            float: The expected returns for the portfolio with the maximum sharpe ratio. 
            float: The expected volatility for the portfolio with the maximum sharpe ratio. 
            float: The sharpe ratio. 
            numpy array: The weights assigned when finding the max sharpe ratio. 
    """
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
    """ Finds the portfolio with the minimum VaR. 

        Args:
            asset_returns (pandas dataframe): Dataframe containing the log returns for each asset. 
            year_days (int): The number of days to be used as a full year.
            c (float): Confidence level for VaR by default set to 2.33 which is 99%. 

        Returns: 
            float: The expected returns for the portfolio with the maximum sharpe ratio. 
            float: The expected volatility for the portfolio with the maximum sharpe ratio. 
            float: The VaR for the portfolio. 
            numpy array: The weights assigned when finding lowest VaR. 

    """
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
    # gets the var for the specified inputs 
    var = var_cov_var(weights, asset_returns, year_days, c)
    return returns, volatility, var, weights

def var_cov_var(asset_weights, asset_returns, year_days, c=2.33):
    """ Calculate the VaR for a portfolio using the covariance variance approach given a set of inputs. 

        Args:
            asset_weights (numpy array): The weights assigned to each asset.
            asset_returns (pandas dataframe): Dataframe containing the log returns for each asset. 
            year_days (int): The number of days to be used as a full year.
            c (float): Confidence level for VaR by default set to 2.33 which is 99%. 

        Returns:
            float: The computed VaR for the specified confidence level. 
    """
    # returns the volatility for the given inputs 
    volatility = annvol_port(asset_weights, asset_returns, year_days) 
    # calculates VaR
    var = c * volatility
    # returns result
    return var

def beta_test_ols(x, y, x_label, y_label):
    """ Performs a beta-test using OLS for two assets. 

        Args:
            x (numpy array): Percentage changes for the first asset. 
            y (numpy array): Percentage changes for the second asset. 
            x_label(str): x-axis label for the plot. 
            y_label(str): y-axis label for the plot. 
    """
    
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
    """ Constructs a binomial tree given the specified inputs 

        Args:
            expected_return (float): The expected return for the asset. 
            expected_volatility (float): The expected volatility for the asset. 
            current_price (float): The current price for the asset. 
            simulation_time (float): The simulation time, 1 = 1yr, 0.5 = 1/2yr and so on. 
            time_periods (int): The number of time periods used to project the prices and probability. 
            verbose (bool): If set to True outputs values during the operation, False will ignore this. By Default set to True. 

        Returns:
            dataframe: Dataframe with the projected prices binomial trees. 
            dataframe: Dataframe with the projected probability binomial trees.
            dataframe: Dataframe with the expected stock prices on the time periods specified.  
    """

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

def arch_vol(returns, year_days, update_freq, verbose=True, **kwargs):
    vol_model = arch_model(returns, kwargs)
    result = vol_model.fit(update_freq=update_freq)
    if verbose:
        print(result.summary())
        
    cond_vol = result.conditional_volatility.iloc[-1]
    annualized_vol = cond_vol * np.sqrt(year_days) * 100
    return annualized_vol

def brownian_motion(n_increments, seed=None):
    """ Creates a brownian path from the increments specified. 

        Args:
            n_increments (int): The time increment. 
            seed (int): A int used to seed the random. 

        Returns:
            numpy array: The brownian path which is the cumulative sum of the brownian increments. 
            numpy array: The brownian increments used in the brownian path. 
    """

    # seed random if passed
    np.random.seed(seed)           
    # get the time step  
    delta_time = 1.0/n_increments  
    # brownian increments
    brownian_increments = np.random.normal(0., 1., int(n_increments)) * np.sqrt(delta_time) 
    # get the brownian path (cumsum of increments)
    brownian_path = np.cumsum(brownian_increments) 
    # return values 
    return brownian_path, brownian_increments

def geo_brownian_motion(current_price, expected_return, expected_volatility, brownian_path, n_increments):    
    """ Simulate prices using geometric brownian motion. 

        Args:
            current_price (float): The price to start simulating from (t = 0). 
            expected_return (float): The expected return for the asset. 
            expected_volatility (float): The expected volatility for the asset. 
            brownian_path (numpy array): The brownian path. 
            n_increment (int) The number of increments to simulate. 
        
        Returns:
            numpy array: The simulated prices over the n_increments. 
            numpy array: The time periods (T0 ... TN). 
    """
    time_periods = np.linspace(0.,1.,int(n_increments + 1))
    stock_prices = []
    stock_prices.append(current_price)
    
    for i in range(1,int(n_increments + 1)):
        drift = (expected_return - 0.5 * expected_volatility ** 2) * time_periods[i]
        diffusion = expected_volatility * brownian_path[i-1]
        tmp_price = current_price * np.exp(drift + diffusion)
        stock_prices.append(tmp_price)
    return stock_prices, time_periods

def rolling_vol(dataframe, window, col=const.COL_LOG_RETURN):
    """ Calculating the rolling volatility for a dataframe. 

        Agrs:
            dataframe (pandas dataframe): The pandas dataframe which has the column used to calculate the volatility.
            window (int): The window used to calculate the rolling volatility. 
            col (str): The column name which will be utilised to calculate the rolling volatility (std).

        Returns:
            pandas series: A series with the rolling volatility. 
    """
    return dataframe[col].rolling(window=window).std()

##########################################################################################################