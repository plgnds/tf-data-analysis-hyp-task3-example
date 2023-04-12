import pandas as pd
import numpy as np


chat_id = 123456 # Ваш chat ID, не меняйте название переменной

def solution(sample_x: np.ndarray, sample_y: np.ndarray = None, condition: int = 1) -> bool:
    if condition == 1:
        alpha = 0.05
        n_x = n_y = 500
    elif condition == 2:
        alpha = 0.0953
        n_x = n_y = 500
    elif condition == 3:
        alpha = 0.0140
        n_x = n_y = 500

    # Calculate the sample means and standard deviations
    x_bar = sample_x.mean()
    y_bar = sample_y.mean() if sample_y is not None else x_bar
    s_x = sample_x.std(ddof=1)
    s_y = sample_y.std(ddof=1) if sample_y is not None else s_x

    # Calculate the pooled standard deviation and test statistic
    s_p = np.sqrt(((n_x - 1) * s_x ** 2 + (n_y - 1) * s_y ** 2) / (n_x + n_y - 2))
    t = (x_bar - y_bar) / (s_p * np.sqrt(1/n_x + 1/n_y))

    # Calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t), df=n_x + n_y - 2))
    if (p_value < alpha):
      return True
    else: 
      return False
