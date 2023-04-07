"""
aioz.aiar.truongle - Nov 23, 2021
init when import package
"""
import os
import logging
import logging.handlers as logHandlers

LOG_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.FATAL
}


LOGGING_TIME_ROTATE = ['S', 'M', 'H', 'D', 'midnight']  # second, minute, hour, day, midnight

# get env
level = os.getenv("LOGGING_LEVEL") if os.getenv("LOGGING_LEVEL") else "DEBUG"
time_rotate = os.getenv("LOGGING_TIME_ROTATE") if os.getenv("LOGGING_TIME_ROTATE") else "midnight"
if time_rotate not in LOGGING_TIME_ROTATE:
    time_rotate = "midnight"
log_file = os.getenv("LOG_FILE_PATH") if os.getenv("LOG_FILE_PATH") else "log/log.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
show_console = True if os.getenv("LOG_SHOW_CONSOLE") in ['true', "True", None] else False
assert level in LOG_LEVEL.keys(), logging.error("Log level {} invalid ... ".format(level))

logger = logging.getLogger('')
logger.setLevel(LOG_LEVEL[level])
# set a format
log_format = '%(asctime)s %(name)-18s %(levelname)-8s :%(message)s'
formatter = logging.Formatter(log_format, datefmt='%Y:%m:%d-%T-%Z%z')

# Roll file
roll_handler = logHandlers.TimedRotatingFileHandler(log_file, when=time_rotate, interval=1, backupCount=2)
roll_handler.setFormatter(formatter)
roll_handler.setLevel(LOG_LEVEL[level])
# logging.basicConfig(level=LOG_LEVEL[level])
logger.addHandler(roll_handler)

# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(LOG_LEVEL[level])
# stdout_handler.setFormatter(formatter)
# logging.getLogger('').addHandler(stdout_handler)

if show_console:
    # define a Handler which writes INFO messages or higher to the sys.stderr
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL[level])
    # tell the handler to use this format
    stream_handler.setFormatter(formatter)
    # add the handler to the root logger
    logger.addHandler(stream_handler)
