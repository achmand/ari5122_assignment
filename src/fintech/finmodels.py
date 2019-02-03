# importing dependencies 
from arch import arch_model

def garch_vol(returns, verbose=True):
    garch = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
    result = garch.fit(update_freq=5)

    if verbose:
        print(result.summary())

    cond_vol = result.conditional_volatility.iloc[-1]
    return cond_vol 

