#
# Define a logger for the scalzi program
#
import logging
from datetime import datetime

def _setup_logger():
    # Create a logger
    logger = logging.getLogger("scalziLogger")
    logger.setLevel(logging.DEBUG)  # Set the logger level to the lowest to capture all messages

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(datetime.now().strftime("logs/scalzi_%Y-%m-%d_%H-%M-%S.log"))

    # Set levels for handlers
    console_handler.setLevel(logging.ERROR)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    file_formatter = logging.Formatter('%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, console_handler, file_handler

scalzi_logger, _console_handler, _file_handler = _setup_logger()

def str_to_log_level(level_str):
    _level = level_str.upper()
    if _level == 'CRITICAL':
        return logging.CRITICAL
    elif _level == "ERROR":
        return logging.ERROR
    elif _level == "WARNING":
        return logging.WARNING
    elif _level == "INFO":
        return logging.INFO

    return logging.CRITICAL+1

def log_level_to_str(level):
    if level == logging.CRITICAL:
        return 'CRITICAL'
    elif level == logging.ERROR:
        return "ERROR"
    elif level == logging.WARNING:
        return "WARNING"
    elif level == logging.INFO:
        return "INFO"

    return "OFF"

def set_file_log_level(level):
    new_level = str_to_log_level(level)
    _file_handler.setLevel(new_level)
    return log_level_to_str(new_level)

def set_console_log_level(level):
    new_level = str_to_log_level(level)
    _console_handler.setLevel(new_level)
    return log_level_to_str(new_level)

