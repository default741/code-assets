# Custom Logging Module Asset to be Used for any Future Projects - v0.0.2

# Author: Abdemanaaf Ghadiali
# Copyright: Copyright 2022, Logging_Module, https://HowToBeBoring.com
# Version: 0.0.2
# Email: abdemanaaf.ghadiali.1998@gmail.com
# Status: Development
# Code Style: PEP8 Style Guide
# MyPy Status: Pass (No Issues)


from pytz import timezone, utc
from pathlib import Path
from time import struct_time

import datetime
import logging
import os
import pytz
import sys


# Default Variables (Changable to Specific Requirements)
__TIMEZONE: str = 'Asia/Kolkata'
__FOLDER_NAME: str = 'logs'
__LEVEL: str = 'info'


# Logging Levels (Code Default: INFO)
__LOGGING_LEVELS: dict = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# Raise Exception if Not a valid Logging Level
if __LEVEL not in __LOGGING_LEVELS:
    raise TypeError(f'Logging Level Not Valid - {__LEVEL}')


def _create_log_folder(folder_name: str = 'logs') -> None:
    """Creates a folder in specified path for Log Files.

    Args:
        folder_name (str, optional): Folder Name. Defaults to 'logs'.
    """
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


_create_log_folder(folder_name=__FOLDER_NAME)


def _logger_custom_time(*args) -> struct_time:
    """Get Current Timezone based on locale for the Logging Module. Localizes UTC time based on Specified
    Timezone and returns the converted Time. Change the timezone to match your requirements. (default: Asia/Kolkata)

    Raises:
        TypeError: If not a valid timezone from pytz.all_timezones.

    Returns:
        struct_time: Current Time based on timezone specified
    """

    if __TIMEZONE not in pytz.all_timezones:
        raise TypeError(f'Not a Valid Timezone - {__TIMEZONE}')

    utc_dt = utc.localize(datetime.datetime.utcnow())
    my_tz = timezone(__TIMEZONE)
    converted = utc_dt.astimezone(my_tz)

    return converted.timetuple()


# Modifying Logger Configuration:
# 1. Format of Logs would be - Time: Root Name: Logging Level: Function Name: Log Message
# 2. Default Logging Level - INFO (Other Levels: DEBUG, WARNING, ERROR, CRITICAL)
# 3. Log Handlers - Default: Save to File and Print to Screen

logging.basicConfig(
    format='%(asctime)s: %(name)s: %(levelname)s: %(funcName)s: %(message)s',
    level=__LOGGING_LEVELS[__LEVEL],
    handlers=[
        logging.FileHandler(Path(
            f'./{__FOLDER_NAME}/logs_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Assign Custom Timezone to Logger Time Converter
logging.Formatter.converter = _logger_custom_time

# Logging Object (Import this object)
# Example: from <relative-path> import logger
logger: logging.Logger = logging.getLogger()
