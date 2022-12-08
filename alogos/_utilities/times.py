import datetime as _datetime


def unix_timestamp_to_readable(unix_timestamp, include_microseconds=False):
    """Convert a UNIX timestamp to a readable format."""
    now = _datetime.datetime.utcfromtimestamp(unix_timestamp)
    if include_microseconds:
        format_str = "%Y-%m-%d %H:%M:%S %f UTC"
    else:
        format_str = "%Y-%m-%d %H:%M:%S UTC"
    return now.strftime(format_str)


def current_time_unix():
    """Get the current time as UNIX timestamp."""
    now_utc = _datetime.datetime.utcnow()
    now_utc = now_utc.replace(tzinfo=_datetime.timezone.utc)
    return now_utc.timestamp()


def current_time_iso(include_microseconds=False):
    """Get the current time as string that is formatted in ISO 8601 standard.

    References
    ----------
    - https://de.wikipedia.org/wiki/ISO_8601
    - https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    - https://stackoverflow.com/questions/2150739/iso-time-iso-8601-in-python

    """
    now_utc = _datetime.datetime.utcnow()
    now_utc = now_utc.replace(tzinfo=_datetime.timezone.utc)
    if not include_microseconds:
        now_utc = now_utc.replace(microsecond=0)
    return now_utc.isoformat()


def current_time_readable(include_microseconds=False):
    """Get the current time as string that is formatted in an easily readable way.

    References
    ----------
    - https://de.wikipedia.org/wiki/ISO_8601
    - https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    - https://stackoverflow.com/questions/2150739/iso-time-iso-8601-in-python

    """
    now_utc = _datetime.datetime.utcnow()
    now_utc = now_utc.replace(tzinfo=_datetime.timezone.utc)
    if include_microseconds:
        format_str = "%Y-%m-%d %H:%M:%S %f UTC"
    else:
        format_str = "%Y-%m-%d %H:%M:%S UTC"
    return now_utc.strftime(format_str)
