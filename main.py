import pandas as pd
import numpy as np
import QuantLib
import json
import os
from tqdm import tqdm
from datetime import datetime as dt
from datetime import time
from dateutil.relativedelta import relativedelta
from py_lib.data.snp_constituents import get_snp500_constituents
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def get_cal():
    nyse_holidays = [dt(x.year(), x.month(), x.dayOfMonth()) for x
                     in QuantLib.UnitedStates(QuantLib.UnitedStates.NYSE).holidayList(
            QuantLib.Date(1, 1, 1901), QuantLib.Date(1, 1, 2100))]

    return pd.offsets.CustomBusinessDay(holidays=nyse_holidays, n=1, weekmask='Mon Tue Wed Thu Fri')


def treat_raw_data(path):
    with open(path, 'r') as f_handle:
        available_data = json.load(f_handle)

    return_dict = {}
    for d in available_data:
        if d['data']:
            return_dict[d['ticker']] = pd.DataFrame(d['data']).set_index('date')['adjClose']

    return pd.concat(return_dict, axis=1).sort_index()


def pre_gen_data(data_start_date, data_end_date, analysis_start_date):
    json_file_path = 'py_lib/data/DailyPricesRaw.json'

    nyse_cal = get_cal()

    df_all = treat_raw_data(json_file_path)
    df_all.index = [dt.combine(x.date(), time(0, 0)) for x in pd.to_datetime(df_all.index)]

    df_log_returns = np.log(df_all.resample(nyse_cal).last()).diff().loc[data_start_date:data_end_date].dropna(how='all', axis=0)

    universe_by_date = pd.concat(
        {
            d: pd.Series({tk: True for tk in get_snp500_constituents(d)})
            for d in tqdm(df_log_returns.index)
        },
        axis=1).T.fillna(False)

    pre_gen_log_returns_hist = []

    sundays = pd.date_range(analysis_start_date, data_end_date, freq=pd.offsets.CustomBusinessDay(weekmask='Sun'))
    rebal_dates = [x - nyse_cal for x in sundays]

    for i, ref_date in enumerate(tqdm(rebal_dates[:-1])):
        ref_date_tickers = universe_by_date.columns[universe_by_date.loc[ref_date]]
        ref_date_tickers = [x for x in ref_date_tickers if x in df_log_returns.columns]

        forward_log_returns = df_log_returns.loc[
                              ref_date + nyse_cal:rebal_dates[i + 1],
                              ref_date_tickers
                              ].dropna(how='all', axis=1).fillna(0)

        forward_effective_ref_date_tickers = list(forward_log_returns.columns)

        hist_log_returns = df_log_returns.loc[
                           ref_date - relativedelta(years=1):ref_date,
                           forward_effective_ref_date_tickers
                           ].dropna(how='all', axis=1).fillna(0)

        effective_ref_date_tickers = list(hist_log_returns.columns)

        pre_gen_log_returns_hist.append({
            '_id': ref_date,
            'date_index': forward_log_returns.index.to_list(),
            'available_tickers': effective_ref_date_tickers,
            'hist_log_returns': hist_log_returns.to_dict('records'),
            'forward_log_returns': forward_log_returns.reindex(effective_ref_date_tickers, axis=1).to_dict('records'),
        })

    return pre_gen_log_returns_hist


def load_price_data(data_start_date, data_end_date, analysis_start_date, force_update=False):
    json_file_path = 'py_lib/data/PriceData.json'

    if force_update or not os.path.exists(json_file_path):
        data = pre_gen_data(data_start_date, data_end_date, analysis_start_date)
        with open(json_file_path, 'w') as f_handle:
            json.dump(data, f_handle, default=str)

    else :
        with open(json_file_path, 'r') as f_handle:
            data = json.load(f_handle)

    return data


def get_max_likelihood_cov_matrix(log_returns):
    from sklearn.covariance import empirical_covariance
    return pd.DataFrame(empirical_covariance(log_returns), index=log_returns.columns, columns=log_returns.columns)


def get_ewma_cov_matrix(log_returns, _lambda=0.94, days_in_a_year=252):
    from itertools import combinations_with_replacement

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

    return ew_covars.combine_first(ew_covars.T)


def get_ledoit_wolf_cov_matrix(log_returns):
    from sklearn.covariance import ledoit_wolf
    covariance, shrinkage = ledoit_wolf(log_returns)
    return pd.DataFrame(covariance, index=log_returns.columns, columns=log_returns.columns)


def get_rp_portfolio(cov_matrix, gross_exposure=1.0):
    n = cov_matrix.shape[0]
    inv_cov = np.linalg.pinv(cov_matrix)  # use pseudo-inverse for safety
    ones = np.ones(n)

    # Step 1: raw inverse risk weights
    raw_weights = inv_cov @ ones

    # Step 2: center to ensure dollar-neutrality
    net_exposure = np.sum(raw_weights)
    centered_weights = raw_weights - (net_exposure / n)

    # Step 3: scale to match gross exposure
    weights = gross_exposure * centered_weights / np.sum(np.abs(centered_weights))

    return {tk: w for tk, w in zip(cov_matrix.columns, weights)}


def correl_dist(corr):
    """Convert correlation matrix to distance matrix."""
    return np.sqrt(0.5 * (1 - corr))


def get_quasi_diag(link):
    link = link.astype(int)
    num_items = link.shape[0] + 1

    def recursively_extract(idx):
        if idx < num_items:
            return [idx]
        else:
            left = link[idx - num_items, 0]
            right = link[idx - num_items, 1]
            return recursively_extract(left) + recursively_extract(right)

    return recursively_extract(len(link) + num_items - 2)


def get_cluster_var(cov, assets):
    """Compute variance of a cluster."""
    sub_cov = cov[np.ix_(assets, assets)]
    w = np.ones(len(assets)) / len(assets)
    return w.T @ sub_cov @ w


def recursive_bisect(cov, sort_ix):
    """Assign weights recursively based on cluster variance."""
    w = pd.Series(1.0, index=sort_ix)

    def _bisect(cov, items):
        if len(items) <= 1:
            return
        split = len(items) // 2
        left = items[:split]
        right = items[split:]

        var_left = get_cluster_var(cov, left)
        var_right = get_cluster_var(cov, right)
        alpha = 1 - var_left / (var_left + var_right)

        w[left] *= alpha
        w[right] *= 1 - alpha

        _bisect(cov, left)
        _bisect(cov, right)

    _bisect(cov, sort_ix)
    return w


def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    return corr


def get_hierarchical_rp_portfolio(cov_matrix, gross_exposure=1.0):
    corr = cov2corr(cov_matrix)
    dist = correl_dist(corr)
    dist_vec = squareform(dist, checks=False)
    link = linkage(dist_vec, method='single')
    sort_ix = get_quasi_diag(link)
    weights = recursive_bisect(cov_matrix.to_numpy(), sort_ix)

    # Recenter to zero net exposure
    centered = weights - weights.mean()

    # Rescale to target gross exposure
    scaled = gross_exposure * centered / np.sum(np.abs(centered))
    index_map = {idx: tk for idx, tk in enumerate(cov_matrix.columns)}
    scaled.index = scaled.index.map(index_map)

    return scaled.reindex(cov_matrix.index).to_dict()


def get_results(weights, returns):
    return (weights * returns).to_dict('records')


def gen_portfolios_hist(start_year, end_year, force_recalc=False):
    json_file_path = f'py_lib/data/Results_{start_year}_{end_year}.json'

    if not force_recalc and os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f_handle:
            try:
                return json.load(f_handle)
            except:
                print("File not readable, recalculating anyway")
                pass

    data_start = dt(start_year-2, 12, 31)
    # DATA_END = dt(2024, 12, 31)
    data_end = dt(end_year, 12, 31)

    analysis_start = dt(start_year, 1, 1)

    data_to_use = load_price_data(data_start, data_end, analysis_start, force_update=True)

    results = []
    for data_point in tqdm(data_to_use):
        ml_cov_matrix = get_max_likelihood_cov_matrix(pd.DataFrame(data_point['hist_log_returns']))
        ewma_cov_matrix = get_ewma_cov_matrix(pd.DataFrame(data_point['hist_log_returns']))
        ledoit_wolf_cov_matrix = get_ledoit_wolf_cov_matrix(pd.DataFrame(data_point['hist_log_returns']))

        ml_cov_matrix_rp_portfolio = get_rp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_rp_portfolio = get_rp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_matrix_rp_portfolio = get_rp_portfolio(ledoit_wolf_cov_matrix)

        ml_cov_matrix_hierarchical_rp_portfolio = get_hierarchical_rp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_hierarchical_rp_portfolio = get_hierarchical_rp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_matrix_hierarchical_rp_portfolio = get_hierarchical_rp_portfolio(ledoit_wolf_cov_matrix)

        equal_weight_portfolio = {t: 1/len(data_point['available_tickers']) for t in data_point['available_tickers']}

        rp_ml_results = get_results(ml_cov_matrix_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        rp_ewma_results = get_results(ewma_cov_matrix_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        rp_ldw_results = get_results(ledoit_wolf_cov_matrix_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        hrp_ml_results = get_results(ml_cov_matrix_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        hrp_ewma_results = get_results(ewma_cov_matrix_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        hrp_ldw_results = get_results(ledoit_wolf_cov_matrix_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        equal_weight_results = get_results(equal_weight_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        results.append(
            data_point | {
                'ml_cov_matrix': ml_cov_matrix.to_dict('records'),
                'ewma_cov_matrix': ewma_cov_matrix.to_dict('records'),
                'ledoit_wolf_cov_matrix': ledoit_wolf_cov_matrix.to_dict('records'),

                'rp_ml_weights': ml_cov_matrix_rp_portfolio,
                'rp_ml_results': rp_ml_results,

                'rp_ewma_weights': ewma_cov_matrix_rp_portfolio,
                'rp_ewma_results': rp_ewma_results,

                'rp_ldw_weights': ledoit_wolf_cov_matrix_rp_portfolio,
                'rp_ldw_results': rp_ldw_results,

                'hrp_ml_weights': ewma_cov_matrix_hierarchical_rp_portfolio,
                'hrp_ml_results': hrp_ml_results,

                'hrp_ewma_weights': ewma_cov_matrix_hierarchical_rp_portfolio,
                'hrp_ewma_results': hrp_ewma_results,

                'hrp_ldw_weights': ledoit_wolf_cov_matrix_hierarchical_rp_portfolio,
                'hrp_ldw_results': hrp_ldw_results,

                'equal_weight_weights': equal_weight_portfolio,
                'equal_weight_results': equal_weight_results,
            }
        )

    with open(json_file_path, 'w') as f_handle:
        json.dump(results, f_handle, default=str)

    return results


def get_daily_matrix_from_hist(data_to_use):
    daily_matrices = {}
    for dt_t_u in tqdm(data_to_use):
        for ref_date in dt_t_u['date_index']:
            daily_matrices[ref_date] = dt_t_u['ewma_cov_matrix']
    return daily_matrices


def risk_contribution(weights, covar_matrix):
    return {1: 1}

def get_concentration_metrics(hist_covar_matrices, hist_allocations):
    hcm_fix = {}
    for d, hcm in hist_covar_matrices.items():
        hcm_df = pd.DataFrame(hcm)
        hcm_df.index = hcm_df.columns
        hcm_fix[d] = hcm_df

    return {
        'risk_allocation': {d: risk_contribution(hist_allocations.loc[d].T.to_dict(), hcm) for d, hcm in hcm_fix.items()},
        'sector_allocation': [],
        'sector_risk_allocation': [],
    }


def get_risk_return_metrics(hist_covar_matrices, hist_returns, freq):
    return {
        'risk_allocation': [],
        'sector_allocation': [],
        'sector_risk_allocation': [],
    }


def evaluate_portfolios(start_year, end_year):
    portfolios_hist = gen_portfolios_hist(start_year, end_year)

    hist_covar_matrices = get_daily_matrix_from_hist(portfolios_hist)

    evaluations = []
    for portfolio_to_evaluate in ['rp_ml', 'rp_ewma', 'rp_ldw_', 'hrp_ml', 'hrp_ewma', 'hrp_ldw_', 'equal_weight',]:
        df_hist_returns = pd.concat([pd.DataFrame(week[f'{portfolio_to_evaluate}_weights'], index=week['date_index'])
                                     for week in portfolios_hist])
        df_hist_allocations = pd.concat([pd.DataFrame(week[f'{portfolio_to_evaluate}_results'], index=week['date_index'])
                                         for week in portfolios_hist])

        concentration_metrics = get_concentration_metrics(hist_covar_matrices, df_hist_allocations)
        weekly_risk_return_metrics = get_risk_return_metrics(hist_covar_matrices, df_hist_returns, 'W')
        monthly_risk_return_metrics = get_risk_return_metrics(hist_covar_matrices, df_hist_returns, 'M')
        year_risk_return_metrics = get_risk_return_metrics(hist_covar_matrices, df_hist_returns, 'Y')

    json_file_path = f'py_lib/data/Evaluation_{start_year}_{end_year}.json'
    with open(json_file_path, 'w') as f_handle:
        json.dump(evaluations, f_handle, default=str)

    return evaluations, portfolios_hist

if __name__ == '__main__':
    evaluation_metrics, hist_data = evaluate_portfolios(2003, 2005)
