from datetime import datetime as dt
import json
import re

today = dt.today().replace(hour=0, minute=0, second=0, microsecond=0)
raw_data_path = './python_code/data/snp_historical_dumb.csv'
target_json_file_path = './python_code/data/snp_add_remove.json'


def treat_snp500_constituents_raw_data(raw_data_file_path):
    with open(raw_data_file_path) as f:
        lines = f.readlines()
    raw_data = [{'date': dt.strptime(line.split(',')[0], '%Y-%m-%d'),
                 'tickers': [re.sub('\s+', '', x.replace('"', ''))
                             for x in line.split(',')[1:]]}
                for line in lines[1:]]

    return raw_data


def get_add_remove_from_raw_data(raw_data_file_path):
    raw_data = treat_snp500_constituents_raw_data(raw_data_file_path)
    add_remove = []
    for i in range(len(raw_data) - 1):
        add = list(set(raw_data[i + 1]['tickers']) - set(raw_data[i]['tickers']))
        remove = list(set(raw_data[i]['tickers']) - set(raw_data[i + 1]['tickers']))
        add_remove.append({'date': raw_data[i + 1]['date'], 'add': add, 'remove': remove})

    return add_remove


def save_add_remove_json_from_raw_data(raw_data_file_path, json_file_path):
    add_remove = [x for x in get_add_remove_from_raw_data(raw_data_file_path) if x['add'] or x['remove']]
    with open(json_file_path, 'w') as f:
        json.dump(add_remove, f, default=str)
