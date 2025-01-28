import math
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement

RANGE = 89
LAMBDA = 0.94
DAYS_IN_A_YEAR = 252
PCT_95_FACTOR = 1.644853627
PCT_99_FACTOR = 2.326347874

def enfusion_var(prices, positions, confidence_factor, window_size=90, _lambda=0.94, days_in_a_year=252):
    SQRT_DAILY_FACTOR = math.sqrt(days_in_a_year) ** -1

    log_returns = np.log(prices).diff().iloc[1:RANGE + 1]
    avg_returns = log_returns.mean()

    square_excess_returns = np.power(log_returns - avg_returns, 2)

    decay = np.power(_lambda, np.array(range(len(square_excess_returns), -1, -1)))[1:]
    decay_sum_inverse = sum(decay) ** -1

    decayed_square_return = (square_excess_returns.T * decay).T

    ew_covars_long = pd.Series({x: (log_returns[x[0]] * log_returns[x[1]] * decay).sum() * decay_sum_inverse * days_in_a_year
               for x in combinations_with_replacement(log_returns.columns, 2)})

    ew_covars = ew_covars_long.reset_index().pivot_table(columns='level_0', index='level_1', values=0)
    for i in decayed_square_return.columns:
        ew_covars.loc[i, i] = decayed_square_return.loc[:, i].sum() * decay_sum_inverse * days_in_a_year
    ew_covars = ew_covars.combine_first(ew_covars.T)

    positions_array = np.array([positions.get(k, 0) for k in ew_covars.columns])

    DAILY_VAR = {i: math.sqrt(ew_covars.loc[i, i])
                          * abs(positions.get(i, 0))
                          * confidence_factor
                          * SQRT_DAILY_FACTOR
                 for i in decayed_square_return.columns}
    daily_incremental_var_array = np.dot(ew_covars, positions_array)
    DAILY_TOTAL_VAR = confidence_factor * SQRT_DAILY_FACTOR * math.sqrt(np.dot(positions_array, daily_incremental_var_array))
    var_contribution_array = (confidence_factor * SQRT_DAILY_FACTOR)**2 * positions_array * daily_incremental_var_array / DAILY_TOTAL_VAR

    return DAILY_TOTAL_VAR, DAILY_VAR, {*zip(ew_covars.columns, var_contribution_array)}


if __name__ == '__main__':
    print('starting')

    FOREX_RATE = 0.128983161

    eg_qtt = {
        'Stock1': 1875400,
        'Stock2': 6290000,
    }

    eg_data = pd.read_csv("D:\Documents\GitHub\RGLib\stock_eg.csv", index_col=0)
    eg_data.index = pd.to_datetime(eg_data.index)

    eg_delta = {k: eg_data.loc[:, k].iloc[-1] * v * FOREX_RATE for k, v in eg_qtt.items()}

    daily_var, total_var, var_contribution = enfusion_var(eg_data, eg_delta, PCT_95_FACTOR, RANGE, LAMBDA, DAYS_IN_A_YEAR)
