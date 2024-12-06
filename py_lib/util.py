from py_lib import settings
from datetime import datetime as dt


today = dt.today().replace(hour=0, minute=0, second=0, microsecond=0)


def treat_date_input(date_input):
    if isinstance(date_input, str):
        try:
            date_input = dt.strptime(date_input, settings.default_date_format)
        except ValueError:
            try:
                date_input = dt.strptime(date_input, settings.default_datetime_format)
            except ValueError:
                raise ValueError(f'date must be a string in the format {settings.default_date_format} '
                                 f'or {settings.default_datetime_format}')

    return date_input
