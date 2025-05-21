import yfinance as yf
import pandas as pd
import QuantLib as ql
import requests
from datetime import datetime as dt
from datetime import timedelta
from tqdm import tqdm
import time
import json
import pickle

from py_lib.data.snp_constituents import get_snp500_constituents
from py_lib.data import snp_constituents

NYSE_holidays = [dt(x.year(), x.month(), x.dayOfMonth()) for x
                 in ql.UnitedStates(ql.UnitedStates.NYSE).holidayList(
        ql.Date(1, 1, 1901), ql.Date(1, 1, 2100))]

NYSE_cal = pd.offsets.CustomBusinessDay(holidays=NYSE_holidays, n=1, weekmask='Mon Tue Wed Thu Fri')

TIINGO_API_KEY_1 = '5345ade70a633f4e7f9fae15b85df135729c8540'
TIINGO_API_KEY_2 = 'be73adc34512bad596ed2b4126ac600eb2fd688c'
TIINGO_API_KEY_3 = '00ece7cc32b7ec36f4196d24e88745f8a03578e5'

# json_file_path = 'py_lib/data/DailyPricesRaw.json'
json_file_path = '.\\DailyPricesRaw.json'

MAX_HOURLY_REQUESTS = 45

DATA_START = '2003-12-31 00:00:00'
DATA_END = '2024-12-31 00:00:00'

relevant_dates = pd.date_range(DATA_START, DATA_END, freq=NYSE_cal)

daily_constituents = pd.concat({
    d: pd.Series({
        tk: True for tk in get_snp500_constituents(d)
    })
    for d in relevant_dates}, axis=1).T.fillna(False)

ticker_date_range = {
    t:
        (
            daily_constituents.index[daily_constituents[t]].min(),
            daily_constituents.index[daily_constituents[t]].max(),
        )
    for t in daily_constituents.columns
}

with open(json_file_path, 'r') as f_handle:
    available_data = json.load(f_handle)

already_got_tickers = [x['ticker'] for x in available_data]

api_keys = [TIINGO_API_KEY_1, TIINGO_API_KEY_2, TIINGO_API_KEY_3]
available_keys = len(api_keys)
key_to_use = 0
requests_made = 0
progress = 0

data_compendium = available_data
tickers_failed = []

posssible_repplaces = {

}

start_time = dt.now()
for ticker, (start_date, end_date) in ticker_date_range.items():
    if ticker not in already_got_tickers:
        ticker = ticker.replace('.', '-')
        api_key = api_keys[key_to_use]

        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        headers = {'Content-Type': 'application/json'}
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "token": api_key
        }

        response = requests.get(url, headers=headers, params=params)

        requests_made += 1
        if requests_made > MAX_HOURLY_REQUESTS or response.status_code == 429:
            key_to_use = (key_to_use + 1) % available_keys
            requests_made = 0
            if key_to_use == 0:
                print("Sleeping to reset hourly requests limit")
                for i in tqdm(range(60 * 60)):
                    time.sleep(1)

        if response.status_code == 200:
            data_compendium.append({
                'ticker': ticker.replace('-', '.'),
                'data': response.json()
            })
        else:
            tickers_failed.append(ticker)

    progress += 1

    if progress % 30 == 0:
        now = dt.now()
        seconds_elapsed = (now - start_time).seconds
        completion_pct = progress / len(ticker_date_range)
        expected_end = now + timedelta(seconds=(seconds_elapsed / completion_pct))
        print(f'{completion_pct * 100:.3} % Done')
        print(f'{seconds_elapsed} Seconds Elapsed')
        print(f'Expected End: {expected_end.strftime('%m-%d %H:%M')}')


with open(json_file_path, 'w') as f_handle:
    json.dump(data_compendium, f_handle, default=str)

