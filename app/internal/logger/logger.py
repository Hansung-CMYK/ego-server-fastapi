import logging
import sys
from logging.config import dictConfig
from textwrap import dedent

dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": dedent("""
            %(asctime)s - %(name)s - %(levelname)s 
            %(message)s
            """),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "level": "INFO",
            "formatter": "default",
        },
    },
    "loggers": {
        "": {                 # root 로거
            "handlers": ["console"],
            "level": "INFO",
        }
    },
})

logger = logging.getLogger(__name__)