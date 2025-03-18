from datetime import datetime as dt
from py_lib import settings
from py_lib import util as pu
import json
import re

#raw_data_path = './py_lib/data/snp_historical_dumb.csv'
raw_data_path = './snp_historical_dumb.csv'
#target_json_file_path = './py_lib/data/snp_add_remove.json'
target_json_file_path = './snp_add_remove.json'


def treat_snp500_constituents_raw_data(raw_data_file_path):
    with open(raw_data_file_path) as f:
        lines = f.readlines()
    raw_data = [{'date': dt.strptime(line.split(',')[0], settings.default_date_format),
                 'tickers': [re.sub(r'\s+', '', x.replace('"', ''))
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


def upsert_snp500_constituents_json(json_file_path, upsert_data):
    """
    Update the S&P 500 constituents json file with the given data
    :param str json_file_path: path to the json file
    :param dict upsert_data: dict with the data to be updated
    :return:
    """
    with open(json_file_path) as f_handle:
        data = json.load(f_handle)

    if 'date' in upsert_data and '_id' in upsert_data:
        if upsert_data['date'] != upsert_data['_id']:
            raise ValueError('update must have a single "date" or "_id" key')
        upsert_data.pop('_id')

    if '_id' in upsert_data:
        upsert_data['date'] = upsert_data.pop('_id')

    assert 'date' in upsert_data, 'update must have a "date" or "_id" key'

    if isinstance(upsert_data['date'], dt):
        upsert_data['date'] = pu.treat_date_input(upsert_data['date']).strftime(settings.default_datetime_format)

    # Check if the update date is already in the data
    dates = [x['date'] for x in data]
    if upsert_data['date'] in dates:
        date_idx = dates.index(upsert_data['date'])
        data[date_idx].update(upsert_data)
    else:
        data.append(upsert_data)
        # Sort the data by date
        data = sorted(data, key=lambda x: x['date'])

    if test_snp500_constituents_json_integrity(data):
        with open(json_file_path, 'w') as f_handle2:
            json.dump(data, f_handle2, default=str)


def test_snp500_constituents_json_integrity(candidate_data):
    """
    Test the integrity of the S&P 500 constituents json data
    :param candidate_data:
    :return:
    """
    for d in candidate_data:
        assert 'date' in d, 'date key not found in data'
        # assert 'add' in d or 'remove' in d, f'no movement for {d["date"]}'
        assert isinstance(d['date'], str), 'date must be a string'
        if not re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', d['date']):
            if re.match(r'\d{4}-\d{2}-\d{2}', d['date']).end() == len(d['date']):
                d['date'] += ' 00:00:00'
            assert re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', d['date']), \
            'date must be in the format YYYY-MM-DD HH:MM:SS'
        try:
            dt.strptime(d['date'], settings.default_datetime_format)
        except ValueError:
            raise AssertionError(f'date must be in the format {settings.default_datetime_format}')
        if 'add' in d:
            assert isinstance(d['add'], list), 'add must be a list'
        if 'remove' in d:
            assert isinstance(d['remove'], list), 'remove must be a list'

    return True


def get_snp500_constituents(ref_date=None, json_file_path=None):
    """
    Get the S&P 500 constituents as of a given date from the json file
    :param ref_date: str or datetime object
    :param json_file_path: str, path to the json file with the S&P 500 constituents changes
    :return:
    """
    if json_file_path is None:
        json_file_path = target_json_file_path

    if ref_date is None:
        ref_date = pu.today

    ref_date = pu.treat_date_input(ref_date)

    with open(json_file_path) as f_handle:
        data = json.load(f_handle)

    assert test_snp500_constituents_json_integrity(data), 'data integrity test failed'
    assert ref_date >= dt.strptime(data[0]['date'],
                                   settings.default_datetime_format), (f'reference date is too early, '
                                                              f'current data starts at {data[0]["date"]}')

    # Cumulative add the "add" field and subtract the "remove" field from the documents
    # that have a date less than or equal to the reference date
    snp_constituents = data[0]['add']
    for d in data[1:]:
        if dt.strptime(d['date'], settings.default_datetime_format) > ref_date:
            break
        snp_constituents = list(set(snp_constituents + d.get('add', [])) - set(d.get('remove', [])))

    return sorted(snp_constituents)


def get_snp500_constituents_changes(start_date, end_date, json_file_path=None):
    """
    Get the changes in the S&P 500 constituents between two dates
    :param start_date: str or datetime object
    :type start_date: str or datetime object
    :param end_date: str or datetime object
    :type end_date: str or datetime object
    :param str json_file_path: str, path to the json file with the S&P 500 constituents changes
    :return: tuple of two lists, the first list contains the tickers added to the S&P 500 between the two dates
                and the second list contains the tickers removed from the S&P 500 between the two dates
    """
    start_date = pu.treat_date_input(start_date)
    end_date = pu.treat_date_input(end_date)

    start_constituents = get_snp500_constituents(start_date, json_file_path)
    end_constituents = get_snp500_constituents(end_date, json_file_path)

    return (list(set(end_constituents) - set(start_constituents)),
            list(set(start_constituents) - set(end_constituents)))


if __name__ == '__main__':
    #TODO get this automated, last check 2024-12-05
    updates_wiki = [
        {'date': '2024-06-24', 'add': ['KKR', 'CRWD', 'GDDY'], 'remove': ['CMA', 'RHI', 'ILMN']},
        {'date': '2024-09-23', 'add': ['DELL', 'PLTR', 'ERIE'], 'remove': ['ETSY', 'BIO', 'AAL']},
        {'date': '2024-09-30', 'add': ['AMTM']},
        {'date': '2024-10-01', 'remove': ['BBWI']},
        {'date': '2024-11-26', 'add': ['TPL'], 'remove': ['MRO']},
        {'date': '2024-12-23', 'add': ['APO', 'WDAY', 'LII'], 'remove': ['QRVO', 'AMTM', 'CTLT']}
        ]

    for update in updates_wiki:
        upsert_snp500_constituents_json(target_json_file_path, update)

    with open(target_json_file_path) as f:
        data = json.load(f)
    test_snp500_constituents_json_integrity(data)

    # Historical year over year change log
    for y in range(1996, pu.today.year + 1): # 1996 is the first year in the data
        added, removed = get_snp500_constituents_changes(f'{y}-01-01', f'{y}-12-31')
        print(f'{y} - added: {len(added)}, removed: {len(removed)}')


    import pandas as pd
    # Current year week over week change log
    for week_start in pd.date_range(start=f'{pu.today.year}-01-01', end=pu.today, freq='W-MON'):
        added, removed = get_snp500_constituents_changes(week_start, week_start + pd.Timedelta(days=6))
        print(f'{week_start.date()} - added: {len(added)}, removed: {len(removed)}')
