import yfinance as yf
import pandas as pd
from datetime import datetime as dt
import QuantLib as ql
import requests

from py_lib.data.snp_constituents import get_snp500_constituents

NYSE_holidays = [dt(x.year(), x.month(), x.dayOfMonth()) for x
                 in ql.UnitedStates(ql.UnitedStates.NYSE).holidayList(
        ql.Date(1, 1, 1901), ql.Date(1, 1, 2100))]

NYSE_cal = pd.offsets.CustomBusinessDay(holidays=NYSE_holidays, n=1, weekmask='Mon Tue Wed Thu Fri')

DATA_START = '2003-12-31 00:00:00'
DATA_END = '2024-12-31 00:00:00'

TIINGO_API_KEY = '00ece7cc32b7ec36f4196d24e88745f8a03578e5'

relevant_dates = pd.date_range(DATA_START, DATA_END, freq=NYSE_cal)

constituents_by_date = {d: pd.Series({x: True for x in get_snp500_constituents(d)}) for d in relevant_dates}

constituents_dense = pd.concat(constituents_by_date,axis=1).T.fillna(False)

date_intervals_per_ticker = {}
for ticker in constituents_dense.columns:
    min_date_ticker = constituents_dense.index[constituents_dense[ticker]].min()
    max_date_ticker = constituents_dense.index[constituents_dense[ticker]].max()

    date_intervals_per_ticker[ticker] = (min_date_ticker, max_date_ticker)

output_data = []

# Loop through each ticker and download the data
for ticker, (start_date, end_date) in date_intervals_per_ticker.items():
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    # Download historical data
    try:
        # df = yf.download(ticker, start=start_date, end=end_date)
        #
        # if not df.empty:
        #     output_data.append({
        #         'ticker': ticker,
        #         'start_date': start_date,
        #         'end_date': end_date,
        #         'data': df.to_dict('records')
        #     })
        # else:

            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            headers = {'Content-Type': 'application/json'}
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "token": TIINGO_API_KEY,
            }

            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                output_data.append({
                    'ticker': ticker,
                    'data': response.json()
                })
            elif response.status_code == 429:  # Too Many Requests
                print(f"Rate limit exceeded for {ticker}. Please try again later.")
                import time
                time.sleep(600)

            else :
                print(f"Failed to fetch data for {ticker} from Tiingo: {response.status_code}")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Save the output data to a file
import json

with open('snp_constituents_price_data.json', 'w') as f:
    json.dump(output_data, f, indent=4)