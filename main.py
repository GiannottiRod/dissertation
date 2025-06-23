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
    import yfinance as yf

    nyse_cal = get_cal()
    snp = yf.download('^GSPC', returns.index[0] - nyse_cal, returns.index[-1] + nyse_cal, auto_adjust=True)

    market_returns = np.log(snp['Close']['^GSPC']).diff().reindex(returns.index)

    return riskmetrics_beta(returns, market_returns)


def adjusted_sharpe_ratio(returns: pd.Series, risk_free_rate: pd.Series, cash_espenditure: pd.Series) -> float:
    from scipy.stats import skew, kurtosis

    r = returns.dropna()

    # Excess returns
    excess = r - risk_free_rate.reindex(r.index) * cash_espenditure.abs()

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


def daily_risk_free_return(start_date: str, end_date: str, api_key='e85ede41d17902f0ae84c45d054d6f6d') -> pd.Series:
    """"Board of Governors of the Federal Reserve System (US), Bank Prime Loan Rate [DPRIME], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DPRIME, June 23, 2025."""
    fred = Fred(api_key=api_key)

    # Step 1: Download DPRIME series
    dprime = fred.get_series('DPRIME', observation_start=start_date, observation_end=end_date)

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

    concentration_metrics = pd.DataFrame(risk_contributions).T.mean().add_suffix('_mean').to_dict() | pd.DataFrame(risk_contributions).T.std().add_suffix('_std').to_dict()

    hist_returns.index = pd.to_datetime(hist_returns.index)
    df_hist_allocations.index = pd.to_datetime(df_hist_allocations.index)

    total_returns = hist_returns.sum(axis=1)
    long_returns = hist_returns[df_hist_allocations > 0].sum(axis=1)
    short_returns = hist_returns[df_hist_allocations < 0].sum(axis=1)

    nyse_cal = get_cal()
    vol = riskmetrics_volatility(total_returns) * np.sqrt(252)
    daily_risk_free = daily_risk_free_return(total_returns.index[0] - nyse_cal, total_returns.index[-1] + nyse_cal)

    rf = (1 + daily_risk_free.reindex(total_returns.index)).prod() - 1

    sharpe = ((np.exp(total_returns - df_hist_allocations.sum(axis=1).abs() * daily_risk_free.reindex(total_returns.index))).prod() - 1) / vol
    adj_sharpe = adjusted_sharpe_ratio(total_returns, daily_risk_free, df_hist_allocations.sum(axis=1))
    total_long_return = np.exp(long_returns.sum()) - 1
    total_short_return = np.exp(short_returns.sum()) - 1

    beta = get_market_beta(total_returns)
    sortino = ((np.exp(total_returns - df_hist_allocations.sum(axis=1).abs() * daily_risk_free.reindex(total_returns.index))).prod() - 1) / abs(beta)

    return {
        'concentration_metrics': concentration_metrics,
    }


def adjust_weights(weights, returns):
    return (weights * np.exp(returns.shift(1).fillna(0))).to_dict('records')


def evaluate_portfolios(start_year, end_year):
    portfolios_hist = gen_portfolios_hist(start_year, end_year)

    hist_covar_matrices = get_daily_matrix_from_hist(portfolios_hist)

    asset_returns = pd.concat([pd.DataFrame(week['forward_log_returns']) for week in portfolios_hist])
    asset_returns.index = pd.to_datetime(list(hist_covar_matrices.keys()))

    evaluations = {}
    for portfolio_to_evaluate in ['rp_ml', 'rp_ewma', 'rp_ldw', 'hrp_ml', 'hrp_ewma', 'hrp_ldw', 'equal_weight',]:
        df_hist_allocations = pd.concat([pd.DataFrame(adjust_weights(week[f'{portfolio_to_evaluate}_weights'],
                                                                     pd.DataFrame(week['forward_log_returns'])),
                                                      index=week['date_index'])
                                     for week in portfolios_hist])
        df_hist_returns = pd.concat([pd.DataFrame(week[f'{portfolio_to_evaluate}_results'], index=week['date_index'])
                                         for week in portfolios_hist])

        metrics = get_metrics(hist_covar_matrices, df_hist_returns, df_hist_allocations, asset_returns)

        evaluations[portfolio_to_evaluate] = metrics

    json_file_path = f'py_lib/data/Evaluation_{start_year}_{end_year}.json'
    with open(json_file_path, 'w') as f_handle:
        json.dump(evaluations, f_handle, default=str)

    return evaluations, portfolios_hist


if __name__ == '__main__':
    start_time = dt.now()
    for y_e in range(2004, 2005):
        print(f'-' * 50)
        print(f'Starting process from {y_e} to {y_e}')
        year_start = dt.now()
        evaluation = evaluate_portfolios(y_e, y_e)
        print(f'Running for {dt.now() - start_time}, last cycle took {dt.now() - year_start}')

