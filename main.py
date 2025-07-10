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
from fredapi import Fred

def get_cal():
    nyse_holidays = [dt(x.year(), x.month(), x.dayOfMonth()) for x
                     in QuantLib.UnitedStates(QuantLib.UnitedStates.NYSE).holidayList(
            QuantLib.Date(1, 1, 1901), QuantLib.Date(1, 1, 2100))]

    return pd.offsets.CustomBusinessDay(holidays=nyse_holidays, n=1, weekmask='Mon Tue Wed Thu Fri')


def get_hist_snp():
    import yfinance as yf

    nyse_cal = get_cal()
    snp_pkl_path = f'py_lib/data/SnPHist.pkl'

    if os.path.exists(snp_pkl_path):
        try:
            return pd.read_pickle(snp_pkl_path)
        except:
            print("File not readable, recalculating anyway")
            pass
    else:
        snp = yf.download('^GSPC', dt(2003, 12, 31) - nyse_cal, dt(2025, 1, 1) + nyse_cal, auto_adjust=True)
        snp.to_pickle(snp_pkl_path)
        return snp


def get_hist_prime_rate(api_key='e85ede41d17902f0ae84c45d054d6f6d'):
    """"Board of Governors of the Federal Reserve System (US), Bank Prime Loan Rate [DPRIME], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DPRIME, June 23, 2025."""
    fred = Fred(api_key=api_key)

    nyse_cal = get_cal()
    dprime_pkl_path = f'py_lib/data/DPrime_hist.pkl'

    if os.path.exists(dprime_pkl_path):
        try:
            return pd.read_pickle(dprime_pkl_path)
        except:
            print("File not readable, recalculating anyway")
            pass
    else:
        dprime = fred.get_series('DPRIME',
                                 observation_start=dt(2003, 12, 31) - nyse_cal,
                                 observation_end=dt(2025, 1, 1) + nyse_cal)
        dprime.to_pickle(dprime_pkl_path)
        return dprime


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
    from sklearn.coscivariance import empirical_covariance
    return pd.DataFrame(empirical_covariance(log_returns), index=log_returns.columns, columns=log_returns.columns)


def get_ewma_cov_matrix_old(log_returns, _lambda=0.94, days_in_a_year=252):
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


def get_ewma_cov_matrix(log_returns: pd.DataFrame, _lambda: float = 0.94, days_in_a_year: int = 252) -> pd.DataFrame:
    from itertools import combinations_with_replacement
    n_obs = len(log_returns)
    assets = log_returns.columns

    # Decay vector: newest to oldest
    decay = np.power(_lambda, np.arange(n_obs - 1, -1, -1))
    decay /= decay.sum()  # normalize weights

    ew_covars = {}

    for i, j in combinations_with_replacement(assets, 2):
        x = log_returns[i].values
        y = log_returns[j].values
        cov_ij = np.sum(decay * (x - x.mean()) * (y - y.mean())) * days_in_a_year
        ew_covars[(i, j)] = cov_ij

    # Convert to matrix
    cov_matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)

    for (i, j), value in ew_covars.items():
        cov_matrix.loc[i, j] = value
        cov_matrix.loc[j, i] = value  # symmetry

    return cov_matrix


def get_ledoit_wolf_cov_matrix(log_returns):
    from sklearn.covariance import ledoit_wolf
    covariance, shrinkage = ledoit_wolf(log_returns)
    return pd.DataFrame(covariance, index=log_returns.columns, columns=log_returns.columns)


def get_rp_portfolio(cov_matrix, gross_exposure=1.0):
    from scipy.optimize import minimize
    n = cov_matrix.shape[0]
    assets = cov_matrix.columns
    Sigma = cov_matrix.values

    # Objective function
    def objective(w):
        w = np.asarray(w)
        var_term = 0.5 * w.T @ Sigma @ w
        entropy_term = - (1 / n) * np.sum(np.log(w))
        return var_term + entropy_term

    # Constraints: sum weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: w_i > 0 (no short-selling)
    bounds = [(1e-8, 1.0) for _ in range(n)]

    # Initial guess: equally weighted
    w0 = np.repeat(1 / n, n)

    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'max_iter': int(10e4)},
    )

    if not result.success:
        # raise RuntimeError(f"Optimization failed: {result.message}")
        return get_ivp_portfolio(cov_matrix, gross_exposure=gross_exposure)

    raw_weights = pd.Series(result.x, index=assets)
    centered_weights = raw_weights - np.mean(raw_weights)
    weights = gross_exposure * centered_weights / np.sum(np.abs(centered_weights))
    return weights.to_dict()


def get_lo_rp_portfolio(cov_matrix, gross_exposure=1.0):
    from scipy.optimize import minimize
    n = cov_matrix.shape[0]
    assets = cov_matrix.columns
    Sigma = cov_matrix.values

    # Objective function
    def objective(w):
        w = np.asarray(w)
        var_term = 0.5 * w.T @ Sigma @ w
        entropy_term = - (1 / n) * np.sum(np.log(w))
        return var_term + entropy_term

    # Constraints: sum weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: w_i > 0 (no short-selling)
    bounds = [(1e-8, 1.0) for _ in range(n)]

    # Initial guess: equally weighted
    w0 = np.repeat(1 / n, n)

    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12}
    )

    if not result.success:
        # raise RuntimeError(f"Optimization failed: {result.message}")
        return get_lo_ivp_portfolio(cov_matrix, gross_exposure=gross_exposure)

    raw_weights = pd.Series(result.x, index=assets)
    weights = gross_exposure * raw_weights / np.sum(np.abs(raw_weights))
    return weights.to_dict()


def get_ivp_portfolio(cov_matrix, gross_exposure=1.0):
    ivp = 1. / np.diag(cov_matrix)
    ivp /= ivp.sum()

    # Step 1: raw inverse risk weights
    raw_weights = ivp

    # Step 2: center to ensure dollar-neutrality
    centered_weights = raw_weights - np.mean(raw_weights)

    # Step 3: scale to match gross exposure
    weights = gross_exposure * centered_weights / np.sum(np.abs(centered_weights))

    return {tk: w for tk, w in zip(cov_matrix.columns, weights)}


def get_lo_ivp_portfolio(cov_matrix, gross_exposure=1.0):
    ivp = 1. / np.diag(cov_matrix)
    ivp /= ivp.sum()

    # Step 1: raw inverse risk weights
    raw_weights = ivp

    # Step 2: scale to match gross exposure
    weights = gross_exposure * raw_weights / np.sum(np.abs(raw_weights))

    return {tk: w for tk, w in zip(cov_matrix.columns, weights)}


def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp


def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx._append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()


def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, int(round(len(i) / 2))),
                                                      (int(round(len(i) / 2)), len(i)))
                  if len(i) > 1]  # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist=np.sqrt((1-corr)/2.)# distance matrix
    return dist


def cov2corr(cov_matrix):
    v = np.sqrt(np.diag(cov_matrix))
    outer_v = np.outer(v, v)
    correl = cov_matrix / outer_v
    correl[cov_matrix == 0] = 0
    return correl


def get_hierarchical_rp_portfolio(cov_matrix, gross_exposure=1.0):
    corr = cov2corr(cov_matrix).round(6)
    corr[corr > 1] = 1
    dist = correlDist(corr)
    link = linkage(dist.to_numpy(), 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    weights = getRecBipart(cov_matrix, sortIx)

    # Recenter to zero net exposure
    centered = weights - weights.mean()

    # Rescale to target gross exposure
    scaled = gross_exposure * centered / np.sum(np.abs(centered))
    # index_map = {idx: tk for idx, tk in enumerate(cov_matrix.columns)}
    # scaled.index = scaled.index.map(index_map)

    return scaled.reindex(cov_matrix.index).to_dict()


def get_longonly_hierarchical_rp_portfolio(cov_matrix, gross_exposure=1.0):
    corr = cov2corr(cov_matrix).round(6)
    corr[corr > 1] = 1
    dist = correlDist(corr)
    link = linkage(dist.to_numpy(), 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    weights = getRecBipart(cov_matrix, sortIx)

    # Rescale to target gross exposure
    scaled = gross_exposure * weights / np.sum(np.abs(weights))

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
        timer_start = dt.now()

        ml_cov_matrix = get_max_likelihood_cov_matrix(pd.DataFrame(data_point['hist_log_returns']))
        ml_end = dt.now()
        print(f'MaxLikelihood Matrix took {ml_end - timer_start}')
        ewma_cov_matrix = get_ewma_cov_matrix(pd.DataFrame(data_point['hist_log_returns']))
        ewma_end = dt.now()
        print(f'EWMA Matrix took {ewma_end - ml_end}')
        ledoit_wolf_cov_matrix = get_ledoit_wolf_cov_matrix(pd.DataFrame(data_point['hist_log_returns']))
        ldw_end = dt.now()
        print(f'Ledoit Wolf Matrix took {ldw_end - ewma_end}')


        ml_cov_matrix_rp_portfolio = get_rp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_rp_portfolio = get_rp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_matrix_rp_portfolio = get_rp_portfolio(ledoit_wolf_cov_matrix)
        rp_end = dt.now()
        print(f'Risk Parity Weights took {rp_end - ldw_end}')

        ml_cov_matrix_longonly_rp_portfolio = get_lo_rp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_longonly_rp_portfolio = get_lo_rp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_matrix_longonly_rp_portfolio = get_lo_rp_portfolio(ledoit_wolf_cov_matrix)
        lo_rp_end = dt.now()
        print(f'LongOnly Risk Parity Weights took {lo_rp_end - rp_end}')

        ml_cov_matrix_ivp_portfolio = get_ivp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_ivp_portfolio = get_ivp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_matrix_ivp_portfolio = get_ivp_portfolio(ledoit_wolf_cov_matrix)
        ivp_end = dt.now()
        print(f'Inverse Variance Weights took {ivp_end - lo_rp_end}')

        ml_cov_matrix_longonly_ivp_portfolio = get_lo_ivp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_longonly_ivp_portfolio = get_lo_ivp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_matrix_longonly_ivp_portfolio = get_lo_ivp_portfolio(ledoit_wolf_cov_matrix)
        lo_ivp_end = dt.now()
        print(f'LongOnly Inverse Variance Weights took {lo_ivp_end - ivp_end}')

        ml_cov_matrix_hierarchical_rp_portfolio = get_hierarchical_rp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_hierarchical_rp_portfolio = get_hierarchical_rp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_matrix_hierarchical_rp_portfolio = get_hierarchical_rp_portfolio(ledoit_wolf_cov_matrix)
        hrp_end = dt.now()
        print(f'Hierarquical Risk Parity Weights took {hrp_end - lo_ivp_end}')

        ml_cov_matrix_longonly_hierarchical_rp_portfolio = get_longonly_hierarchical_rp_portfolio(ml_cov_matrix)
        ewma_cov_matrix_longonly_hierarchical_rp_portfolio = get_longonly_hierarchical_rp_portfolio(ewma_cov_matrix)
        ledoit_wolf_cov_longonly_matrix_hierarchical_rp_portfolio = get_longonly_hierarchical_rp_portfolio(ledoit_wolf_cov_matrix)
        lo_hrp_end = dt.now()
        print(f'LongOnly Hierarquical Risk Parity Weights took {lo_hrp_end - hrp_end}')

        equal_weight_portfolio = {t: 1/len(data_point['available_tickers']) for t in data_point['available_tickers']}

        rp_ml_results = get_results(ml_cov_matrix_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        rp_ewma_results = get_results(ewma_cov_matrix_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        rp_ldw_results = get_results(ledoit_wolf_cov_matrix_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        lo_rp_ml_results = get_results(ml_cov_matrix_longonly_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        lo_rp_ewma_results = get_results(ewma_cov_matrix_longonly_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        lo_rp_ldw_results = get_results(ledoit_wolf_cov_matrix_longonly_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        ivp_ml_results = get_results(ml_cov_matrix_ivp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        ivp_ewma_results = get_results(ewma_cov_matrix_ivp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        ivp_ldw_results = get_results(ledoit_wolf_cov_matrix_ivp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        lo_ivp_ml_results = get_results(ml_cov_matrix_longonly_ivp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        lo_ivp_ewma_results = get_results(ewma_cov_matrix_longonly_ivp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        lo_ivp_ldw_results = get_results(ledoit_wolf_cov_matrix_longonly_ivp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        hrp_ml_results = get_results(ml_cov_matrix_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        hrp_ewma_results = get_results(ewma_cov_matrix_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        hrp_ldw_results = get_results(ledoit_wolf_cov_matrix_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        lo_hrp_ml_results = get_results(ml_cov_matrix_longonly_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        lo_hrp_ewma_results = get_results(ewma_cov_matrix_longonly_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))
        lo_hrp_ldw_results = get_results(ledoit_wolf_cov_longonly_matrix_hierarchical_rp_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        equal_weight_results = get_results(equal_weight_portfolio, pd.DataFrame(data_point['forward_log_returns']))

        results_end = dt.now()
        print(f'Results took {results_end - lo_hrp_end}')

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

                'lo_rp_ml_weights': ml_cov_matrix_longonly_rp_portfolio,
                'lo_rp_ml_results': lo_rp_ml_results,

                'lo_rp_ewma_weights': ewma_cov_matrix_longonly_rp_portfolio,
                'lo_rp_ewma_results': lo_rp_ewma_results,

                'lo_rp_ldw_weights': ledoit_wolf_cov_matrix_longonly_rp_portfolio,
                'lo_rp_ldw_results': lo_rp_ldw_results,

                'ivp_ml_weights': ml_cov_matrix_ivp_portfolio,
                'ivp_ml_results': ivp_ml_results,

                'ivp_ewma_weights': ewma_cov_matrix_ivp_portfolio,
                'ivp_ewma_results': ivp_ewma_results,

                'ivp_ldw_weights': ledoit_wolf_cov_matrix_ivp_portfolio,
                'ivp_ldw_results': ivp_ldw_results,

                'lo_ivp_ml_weights': ml_cov_matrix_longonly_ivp_portfolio,
                'lo_ivp_ml_results': lo_ivp_ml_results,

                'lo_ivp_ewma_weights': ewma_cov_matrix_longonly_ivp_portfolio,
                'lo_ivp_ewma_results': lo_ivp_ewma_results,

                'lo_ivp_ldw_weights': ledoit_wolf_cov_matrix_longonly_ivp_portfolio,
                'lo_ivp_ldw_results': lo_ivp_ldw_results,

                'hrp_ml_weights': ewma_cov_matrix_hierarchical_rp_portfolio,
                'hrp_ml_results': hrp_ml_results,

                'hrp_ewma_weights': ewma_cov_matrix_hierarchical_rp_portfolio,
                'hrp_ewma_results': hrp_ewma_results,

                'hrp_ldw_weights': ledoit_wolf_cov_matrix_hierarchical_rp_portfolio,
                'hrp_ldw_results': hrp_ldw_results,

                'lo_hrp_ml_weights': ewma_cov_matrix_longonly_hierarchical_rp_portfolio,
                'lo_hrp_ml_results': lo_hrp_ml_results,

                'lo_hrp_ewma_weights': ewma_cov_matrix_longonly_hierarchical_rp_portfolio,
                'lo_hrp_ewma_results': lo_hrp_ewma_results,

                'lo_hrp_ldw_weights': ledoit_wolf_cov_longonly_matrix_hierarchical_rp_portfolio,
                'lo_hrp_ldw_results': lo_hrp_ldw_results,

                'equal_weight_weights': equal_weight_portfolio,
                'equal_weight_results': equal_weight_results,
            }
        )

    export_start = dt.now()
    with open(json_file_path, 'w') as f_handle:
        json.dump(results, f_handle, default=str)
    print(f'Export took {dt.now() - export_start}')

    return results


def get_daily_matrix_from_hist(data_to_use):
    daily_matrices = {}
    for dt_t_u in tqdm(data_to_use):
        for ref_date in dt_t_u['date_index']:
            daily_matrices[ref_date] = dt_t_u['ewma_cov_matrix']
    return daily_matrices


def risk_contribution(weights_dict, covar_matrix):
    # Ensure assets in the same order
    assets = covar_matrix.columns
    weights = np.array([weights_dict[asset] for asset in assets])
    covar_matrix = covar_matrix.values

    # Portfolio volatility
    portfolio_var = weights.T @ covar_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_var)

    # Marginal contribution to risk
    marginal_contrib = (covar_matrix @ weights) / portfolio_volatility

    # Total risk contribution
    risk_contribution_ = {k: v for k, v in zip(assets, weights * marginal_contrib)}

    risk_contribution_df = pd.Series(risk_contribution_)
    weights_dict_df = pd.Series(weights_dict)

    contributions = {
        'parametrical_var': risk_contribution_df.sum(),
        'net_expo': weights_dict_df.sum(),
        'abs_net_expo': abs(weights_dict_df.sum()),
        'gross_expo': weights_dict_df.abs().sum(),
        'long_risk_contribution': risk_contribution_df[weights_dict_df > 0].sum(),
        'long_money_contribution': weights_dict_df[weights_dict_df > 0].sum(),
        'short_risk_contribution': risk_contribution_df[weights_dict_df < 0].sum(),
        'short_money_contribution': weights_dict_df[weights_dict_df < 0].sum(),
    }
    for pct in [0.01, 0.05, 0.1, 0.25]:
        n_relevant = round(pct * len(assets))
        contributions[f'risk_conc_at_{int(pct*100)}pct'] = risk_contribution_df.abs().sort_values().iloc[
                                                           -n_relevant:].sum() / risk_contribution_df.abs().sum()
        contributions[f'money_conc_at_{int(pct * 100)}pct'] = weights_dict_df.abs().sort_values().iloc[
                                                             -n_relevant:].sum() / weights_dict_df.abs().sum()
    return contributions


def get_risk_free_rate(start_date, end_date, ticker="BIL"):
    import yfinance as yf

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close']
    returns = prices.pct_change().dropna()
    return returns


def riskmetrics_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    returns = returns.dropna()
    ewma_var = [returns.var()]

    for t in range(1, len(returns)):
        var_t = lambda_ * ewma_var[-1] + (1 - lambda_) * returns.iloc[t - 1] ** 2
        ewma_var.append(var_t)

    volatility = np.sqrt(pd.Series(ewma_var, index=returns.index))
    return volatility.iloc[-1]


def riskmetrics_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    data = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    r_a = data.iloc[:, 0]
    r_b = data.iloc[:, 1]

    var_b = [r_b.var()]
    cov_ab = [np.cov(r_a, r_b)[0, 1]]

    for t in range(1, len(data)):
        var_b_t = lambda_ * var_b[-1] + (1 - lambda_) * r_b.iloc[t - 1] ** 2
        cov_ab_t = lambda_ * cov_ab[-1] + (1 - lambda_) * r_a.iloc[t - 1] * r_b.iloc[t - 1]
        var_b.append(var_b_t)
        cov_ab.append(cov_ab_t)

    beta = pd.Series(cov_ab, index=data.index) / pd.Series(var_b, index=data.index)
    return beta.iloc[-1]


def get_market_beta(returns):
    snp = get_hist_snp()

    market_returns = np.log(snp['Close']['^GSPC']).diff().reindex(returns.index)

    return riskmetrics_beta(returns, market_returns)


def adjusted_sharpe_ratio(returns: pd.Series, risk_free_rate: pd.Series, cash_espenditure: pd.Series) -> float:
    from scipy.stats import skew, kurtosis

    r = returns.dropna()

    # Excess returns
    excess = r - risk_free_rate.reindex(r.index) * cash_espenditure.reindex(r.index).abs()

    # Mean and standard deviation
    mean_excess = excess.mean()
    std_dev = excess.std(ddof=0)
    sr = mean_excess / std_dev

    # Higher moments
    s = skew(excess)
    k = kurtosis(excess, fisher=False)  # Use non-excess kurtosis (normal dist = 3)

    # Adjusted Sharpe formula
    adj_sr = sr * (1 + (s / 6) * sr - ((k - 3) / 24) * sr**2)

    adj_sr *= np.sqrt(len(r))

    return adj_sr


def daily_risk_free_return(start_date: str, end_date: str) -> pd.Series:
    # Step 1: Download/Retrieve DPRIME series
    dprime = get_hist_prime_rate()

    # Step 2: Interpolate to daily frequency
    daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
    dprime_daily = dprime.reindex(daily_index).interpolate(method='linear')

    # Step 3: Convert annualized % rate to daily decimal return (assuming 252 trading days)
    daily_rf = (dprime_daily / 100) / 252

    return daily_rf


def get_metrics(hist_covar_matrices, hist_returns, df_hist_allocations, asset_returns):
    hcm_fix = {}
    for d, hcm in hist_covar_matrices.items():
        hcm_df = pd.DataFrame(hcm)
        hcm_df.index = hcm_df.columns
        hcm_fix[d] = hcm_df

    risk_contributions = {d: risk_contribution(df_hist_allocations.loc[d].T.fillna(0).to_dict(), hcm) for d, hcm in hcm_fix.items()}

    risk_df = pd.DataFrame(risk_contributions).T
    risk_df.index = pd.to_datetime(risk_df.index)

    concentration_metrics_y = (risk_df.mean().add_suffix('_mean').to_dict() |
                               risk_df.std().add_suffix('_std').to_dict())
    concentration_metrics_m = pd.concat([risk_df.resample('ME').mean().add_suffix('_mean'),
                                         risk_df.resample('ME').std().add_suffix('_std')], axis=1)

    hist_returns.index = pd.to_datetime(hist_returns.index)
    df_hist_allocations.index = pd.to_datetime(df_hist_allocations.index)

    total_returns_d = hist_returns.sum(axis=1)
    long_returns_d = hist_returns[df_hist_allocations > 0].sum(axis=1)
    short_returns_d = hist_returns[df_hist_allocations < 0].sum(axis=1)

    nyse_cal = get_cal()
    vol_y = riskmetrics_volatility(total_returns_d) * np.sqrt(252)
    vol_m = total_returns_d.resample('ME').apply(lambda x: riskmetrics_volatility(x) * np.sqrt(21))
    daily_risk_free = daily_risk_free_return(total_returns_d.index[0] - nyse_cal, total_returns_d.index[-1] + nyse_cal)

    rf_y = (1 + daily_risk_free.reindex(total_returns_d.index)).prod() - 1
    rf_m = (1 + daily_risk_free.reindex(total_returns_d.index)).resample('ME').prod()

    daily_cash_expend = df_hist_allocations.sum(axis=1).abs()

    sharpe_y = ((np.exp(total_returns_d - daily_cash_expend * daily_risk_free.reindex(total_returns_d.index))).prod() - 1) / vol_y
    sharpe_m = ((np.exp(total_returns_d - daily_cash_expend * daily_risk_free.reindex(total_returns_d.index))).resample('ME').prod() - 1) / vol_m

    adj_sharpe_y = adjusted_sharpe_ratio(total_returns_d, daily_risk_free, daily_cash_expend)
    adj_sharpe_m = total_returns_d.resample('ME').apply(lambda x: adjusted_sharpe_ratio(x, daily_risk_free, daily_cash_expend))

    total_return_y = np.exp(total_returns_d.sum()) - 1
    total_return_m = np.exp(total_returns_d.resample('ME').sum()) - 1

    total_long_return_y = np.exp(long_returns_d.sum()) - 1
    total_long_return_m = np.exp(long_returns_d.resample('ME').sum()) - 1

    total_short_return_y = np.exp(short_returns_d.sum()) - 1
    total_short_return_m = np.exp(short_returns_d.resample('ME').sum()) - 1

    beta_y = get_market_beta(total_returns_d)
    beta_m = total_returns_d.resample('ME').apply(lambda x: get_market_beta(x))

    sortino_y = ((np.exp(total_returns_d - daily_cash_expend * daily_risk_free.reindex(
        total_returns_d.index))).prod() - 1) / abs(beta_y)
    sortino_m = ((np.exp(total_returns_d - daily_cash_expend * daily_risk_free.reindex(
        total_returns_d.index))).resample('ME').prod() - 1) / abs(beta_m)

    yearly_metrics = {hist_returns.index[0].replace(day=1).strftime('%Y-%m-%d'): {
                          'sharpe': sharpe_y,
                          'adj_sharpe': adj_sharpe_y,
                          'total_return': total_return_y,
                          'total_long_return': total_long_return_y,
                          'total_short_return': total_short_return_y,
                          'beta': beta_y,
                          'sortino': sortino_y,
                          'risk_free': rf_y,
                          'vol': vol_y,
                      } | concentration_metrics_y}

    monthly_metrics = pd.concat([pd.concat(
        {
             'sharpe': sharpe_m,
             'adj_sharpe': adj_sharpe_m,
             'total_return': total_return_m,
             'total_long_return': total_long_return_m,
             'total_short_return': total_short_return_m,
             'beta': beta_m,
             'sortino': sortino_m,
             'risk_free': rf_m,
             'vol': vol_m,
        }, axis=1),
        concentration_metrics_m
    ], axis=1)

    return yearly_metrics, monthly_metrics

def get_metrics_w(hist_covar_matrices, hist_returns, df_hist_allocations, asset_returns):
    hcm_fix = {}
    for d, hcm in hist_covar_matrices.items():
        hcm_df = pd.DataFrame(hcm)
        hcm_df.index = hcm_df.columns
        hcm_fix[d] = hcm_df

    risk_contributions = {d: risk_contribution(df_hist_allocations.loc[d].T.fillna(0).to_dict(), hcm) for d, hcm in hcm_fix.items()}

    risk_df = pd.DataFrame(risk_contributions).T
    risk_df.index = pd.to_datetime(risk_df.index)

    concentration_metrics_w = pd.concat([risk_df.resample('W').mean().add_suffix('_mean'),
                                         risk_df.resample('W').std().add_suffix('_std')], axis=1)

    hist_returns.index = pd.to_datetime(hist_returns.index)
    df_hist_allocations.index = pd.to_datetime(df_hist_allocations.index)

    total_returns_d = hist_returns.sum(axis=1)
    long_returns_d = hist_returns[df_hist_allocations > 0].sum(axis=1)
    short_returns_d = hist_returns[df_hist_allocations < 0].sum(axis=1)

    nyse_cal = get_cal()
    vol_w = total_returns_d.resample('W').apply(lambda x: riskmetrics_volatility(x) * np.sqrt(21))
    daily_risk_free = daily_risk_free_return(total_returns_d.index[0] - nyse_cal, total_returns_d.index[-1] + nyse_cal)

    rf_w = (1 + daily_risk_free.reindex(total_returns_d.index)).resample('W').prod()

    daily_cash_expend = df_hist_allocations.sum(axis=1).abs()

    sharpe_w = ((np.exp(total_returns_d - daily_cash_expend * daily_risk_free.reindex(total_returns_d.index))).resample('W').prod() - 1) / vol_w

    adj_sharpe_w = total_returns_d.resample('W').apply(lambda x: adjusted_sharpe_ratio(x, daily_risk_free, daily_cash_expend))

    total_return_w = np.exp(total_returns_d.resample('W').sum()) - 1

    total_long_return_w = np.exp(long_returns_d.resample('W').sum()) - 1

    total_short_return_w = np.exp(short_returns_d.resample('W').sum()) - 1

    beta_w = total_returns_d.resample('W').apply(lambda x: get_market_beta(x))

    sortino_w = ((np.exp(total_returns_d - daily_cash_expend * daily_risk_free.reindex(
        total_returns_d.index))).resample('W').prod() - 1) / abs(beta_w)

    monthly_metrics = pd.concat([pd.concat(
        {
             'sharpe': sharpe_w,
             'adj_sharpe': adj_sharpe_w,
             'total_return': total_return_w,
             'total_long_return': total_long_return_w,
             'total_short_return': total_short_return_w,
             'beta': beta_w,
             'sortino': sortino_w,
             'risk_free': rf_w,
             'vol': vol_w,
        }, axis=1),
        concentration_metrics_w
    ], axis=1)

    return monthly_metrics


def test_covariance_matrix(S, Sigma_0, n, alpha=0.05):
    from scipy.stats import chi2

    p = S.shape[0]

    # Compute test statistic
    log_det_Sigma0 = np.linalg.slogdet(Sigma_0)[1]
    log_det_S = np.linalg.slogdet(S)[1]
    inv_Sigma0 = np.linalg.pinv(Sigma_0)

    trace_term = np.trace(inv_Sigma0 @ S)

    T = n * (log_det_Sigma0 - log_det_S + trace_term - p)

    # Degrees of freedom
    df = int(p * (p + 1) / 2)

    # Critical value and p-value
    critical_value = chi2.ppf(1 - alpha, df)
    p_value = 1 - chi2.cdf(T, df)

    return {
        'test_statistic': T,
        'critical_value': critical_value,
        'p_value': p_value,
        'reject_H0': T > critical_value
    }


def matrix_mse(estimated: pd.DataFrame, observed: pd.DataFrame):
    # Align matrices
    est = estimated.values
    obs = observed.values

    if est.shape != obs.shape:
        raise ValueError("Matrices must be the same shape.")

    # Flatten upper triangle (including diagonal)
    triu_indices = np.triu_indices_from(est)
    est_flat = est[triu_indices]
    obs_flat = obs[triu_indices]

    squared_errors = (est_flat - obs_flat) ** 2

    return np.mean(squared_errors)


def matrix_medae(estimated: pd.DataFrame, observed: pd.DataFrame):
    # Align matrices
    est = estimated.values
    obs = observed.values

    if est.shape != obs.shape:
        raise ValueError("Matrices must be the same shape.")

    # Flatten upper triangle (including diagonal)
    triu_indices = np.triu_indices_from(est)
    est_flat = est[triu_indices]
    obs_flat = obs[triu_indices]

    absolute_errors = np.abs(est_flat - obs_flat)

    return np.median(absolute_errors)


def cvmatrix_evaluations(estimated_matrix, asset_returns):
    estimated_matrix.index = estimated_matrix.columns
    realized_cvar = asset_returns.cov()
    realized_cvar = realized_cvar.reindex(estimated_matrix.index).reindex(estimated_matrix.columns, axis=1)
    return {
        'mse': matrix_mse(estimated_matrix, realized_cvar),
        'medae': matrix_medae(estimated_matrix, realized_cvar),
        'ttest': test_covariance_matrix(estimated_matrix,
                                        realized_cvar,
                                        asset_returns.shape[0] * asset_returns.shape[1]
                                        ),
    }


def adjust_weights(weights, returns):
    return (weights * np.exp(returns.shift(1).fillna(0))).to_dict('records')


def evaluate_portfolios(start_year, end_year, force_recalc):
    portfolios_hist = gen_portfolios_hist(start_year, end_year, force_recalc)

    json_file_path_w = f'py_lib/data/Evaluation_w_{start_year}_{end_year}.json'
    json_file_path_m = f'py_lib/data/Evaluation_m_{start_year}_{end_year}.json'
    json_file_path_y = f'py_lib/data/Evaluation_y_{start_year}_{end_year}.json'
    json_file_path_cvarmx = f'py_lib/data/Evaluation_cvarmx_{start_year}_{end_year}.json'

    if not force_recalc and all([os.path.exists(json_file_path_y),
                                 os.path.exists(json_file_path_m),
                                 os.path.exists(json_file_path_w),
                                 os.path.exists(json_file_path_cvarmx),]):
        try:
            with (open(json_file_path_y, 'r') as f_handle_y):
                eval_year = json.load(f_handle_y)
            with (open(json_file_path_m, 'r') as f_handle_m):
                eval_month = json.load(f_handle_m)
            with (open(json_file_path_w, 'r') as f_handle_w):
                eval_week = json.load(f_handle_w)
            with (open(json_file_path_cvarmx, 'r') as f_handle_cvarmx):
                eval_cvarmx = json.load(f_handle_cvarmx)
            return eval_year, eval_month, eval_week, eval_cvarmx, portfolios_hist
        except:
            print("Files not readable, recalculating anyway")
            pass

    hist_covar_matrices = get_daily_matrix_from_hist(portfolios_hist)

    asset_returns = pd.concat([pd.DataFrame(week['forward_log_returns']) for week in portfolios_hist])
    asset_returns.index = pd.to_datetime(list(hist_covar_matrices.keys()))

    if not force_recalc and os.path.exists(json_file_path_cvarmx):
        with (open(json_file_path_cvarmx, 'r') as f_handle_cvarmx):
            eval_cvarmx = json.load(f_handle_cvarmx)

    else:
        eval_cvarmx = [{
            'index': week['date_index'][0],
            'ml_cov_matrix': cvmatrix_evaluations(pd.DataFrame(week['ml_cov_matrix']), pd.DataFrame(week['forward_log_returns'])),
            'ewma_cov_matrix': cvmatrix_evaluations(pd.DataFrame(week['ewma_cov_matrix']), pd.DataFrame(week['forward_log_returns'])),
            'ledoit_wolf_cov_matrix': cvmatrix_evaluations(pd.DataFrame(week['ledoit_wolf_cov_matrix']), pd.DataFrame(week['forward_log_returns'])),
        } for week in portfolios_hist]

        with (open(json_file_path_cvarmx, 'w') as f_handle_cvarmx):
            json.dump(eval_cvarmx, f_handle_cvarmx, default=str)

    if not force_recalc and os.path.exists(json_file_path_w):
        with (open(json_file_path_w, 'r') as f_handle_w):
            evaluations_w_fix = json.load(f_handle_w)

    else:
        evaluations_w = {}
        for portfolio_to_evaluate in tqdm([
            'ivp_ml', 'ivp_ewma', 'ivp_ldw',
            'lo_ivp_ml', 'lo_ivp_ewma', 'lo_ivp_ldw',
            'rp_ml', 'rp_ewma', 'rp_ldw',
            'lo_rp_ml', 'lo_rp_ewma', 'lo_rp_ldw',
            'hrp_ml', 'hrp_ewma', 'hrp_ldw',
            'lo_hrp_ml', 'lo_hrp_ewma', 'lo_hrp_ldw',
            'equal_weight',
        ]):
            df_hist_allocations = pd.concat([pd.DataFrame(adjust_weights(week[f'{portfolio_to_evaluate}_weights'],
                                                                         pd.DataFrame(week['forward_log_returns'])),
                                                          index=week['date_index'])
                                             for week in portfolios_hist])
            df_hist_returns = pd.concat(
                [pd.DataFrame(week[f'{portfolio_to_evaluate}_results'], index=week['date_index'])
                 for week in portfolios_hist])

            metrics_w = get_metrics_w(hist_covar_matrices, df_hist_returns, df_hist_allocations, asset_returns)

            evaluations_w[portfolio_to_evaluate] = metrics_w

        evaluations_w_fix = pd.concat(evaluations_w).reset_index().rename(
            columns={'level_0': 'portfolio', 'level_1': 'date'}).to_dict('records')
        with (open(json_file_path_w, 'w') as f_handle_w):
            json.dump(evaluations_w_fix, f_handle_w, default=str)

    if not force_recalc and all([os.path.exists(json_file_path_y),
                                 os.path.exists(json_file_path_m),]):
        with (open(json_file_path_y, 'r') as f_handle_y):
            evaluations_y = json.load(f_handle_y)
        with (open(json_file_path_m, 'r') as f_handle_m):
            evaluations_m_fix = json.load(f_handle_m)

    else:
        evaluations_y = {}
        evaluations_m = {}
        for portfolio_to_evaluate in [
            'ivp_ml', 'ivp_ewma', 'ivp_ldw',
            'lo_ivp_ml', 'lo_ivp_ewma', 'lo_ivp_ldw',
            'rp_ml', 'rp_ewma', 'rp_ldw',
            'lo_rp_ml', 'lo_rp_ewma', 'lo_rp_ldw',
            'hrp_ml', 'hrp_ewma', 'hrp_ldw',
            'lo_hrp_ml', 'lo_hrp_ewma', 'lo_hrp_ldw',
            'equal_weight',
        ]:
            df_hist_allocations = pd.concat([pd.DataFrame(adjust_weights(week[f'{portfolio_to_evaluate}_weights'],
                                                                         pd.DataFrame(week['forward_log_returns'])),
                                                          index=week['date_index'])
                                         for week in portfolios_hist])
            df_hist_returns = pd.concat([pd.DataFrame(week[f'{portfolio_to_evaluate}_results'], index=week['date_index'])
                                             for week in portfolios_hist])

            metrics_y, metrics_m = get_metrics(hist_covar_matrices, df_hist_returns, df_hist_allocations, asset_returns)

            evaluations_y[portfolio_to_evaluate] = metrics_y
            evaluations_m[portfolio_to_evaluate] = metrics_m

        evaluations_m_fix = pd.concat(evaluations_m).reset_index().rename(
            columns={'level_0': 'portfolio', 'level_1': 'date'}).to_dict('records')

        with open(json_file_path_y, 'w') as f_handle:
            json.dump(evaluations_y, f_handle, default=str)

        with open(json_file_path_m, 'w') as f_handle:
            json.dump(evaluations_m_fix, f_handle, default=str)

    return evaluations_y, pd.DataFrame(evaluations_m_fix), pd.DataFrame(evaluations_w_fix), eval_cvarmx, portfolios_hist


if __name__ == '__main__':
    rerun = False

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message='The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix')

    start_time = dt.now()

    evaluations_y = []
    evaluations_m = []
    evaluations_w = []
    portfolios_hist = []
    eval_cvarmx_hist = []
    for y_e in range(2004, 2025):
        print(f'-' * 50)
        print(f'Starting process from {y_e} to {y_e}')
        year_start = dt.now()
        eval_y, eval_m, eval_w, eval_cvarmx, port_hist = evaluate_portfolios(y_e, y_e, force_recalc=rerun)
        evaluations_y.append(eval_y)
        evaluations_m.append(eval_m)
        evaluations_w.append(eval_w)
        eval_cvarmx_hist.append(eval_cvarmx)
        portfolios_hist.append(port_hist)
        print(f'Running for {dt.now() - start_time}, last cycle took {dt.now() - year_start}')
    print('Done with generating data')
    cvmx_eval_df_raw = pd.concat([pd.concat(
        {w['index']: pd.concat({k: pd.Series(v) for k, v in w.items() if k != 'index'}, axis=1).T for w in y}) for y
                                  in eval_cvarmx_hist])

    (
        pd.concat([pd.concat({k: pd.DataFrame(v) for k, v in x.items()}).T for x in evaluations_y]).reorder_levels([1, 0], 1),
        pd.concat([pd.DataFrame(x) for x in evaluations_m]).pivot_table(index='date', columns='portfolio'),
        pd.concat([pd.DataFrame(x) for x in evaluations_w]).pivot_table(index='date', columns='portfolio'),
        cvmx_eval_df_raw,
        portfolios_hist,
        pd.concat([pd.Series({c['date_index'][0]: len(c['available_tickers']) for c in h}) for h in portfolios_hist])
    )
