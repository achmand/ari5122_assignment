# importing dependencies 
import numpy as np 
from arch import arch_model
from scipy.stats import skew
from scipy.stats import kurtosis
import statsmodels.api as sm
import matplotlib.pyplot as plt 

def dist_moments(x):
    return np.mean(x), np.std(x), skew(x), kurtosis(x)

def annualized_retvol(first_mom, second_mom, year_days):
    year_return = first_mom * year_days
    year_vol = second_mom * np.sqrt(year_days) 

    annualized_return = year_return * 100
    annualized_volatility = year_vol * 100

    return annualized_return, annualized_volatility

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
    fig = plt.figure(figsize=(10,7))
    plt.scatter(x, y, alpha=.3)
    plt.xlabel(x_label + " Daily Return")
    plt.ylabel(y_label + " Daily Return")
    plt.plot(x2, pred_y, "r", alpha = 0.9)
    plt.show()

