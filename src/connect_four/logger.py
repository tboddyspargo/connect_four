from enum import IntEnum, auto
from traceback import format_exc
from typing import Union


class LogLevel(IntEnum):
    NONE = auto()
    INFO = auto()
    DEBUG = auto()
    VERBOSE = auto()

    def __ge__(self, other: 'LogLevel') -> bool:
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other: 'LogLevel') -> bool:
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other: 'LogLevel') -> bool:
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other: 'LogLevel') -> bool:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Logger:
    """Logger is a class for logging messages to the console host with consistent formatting."""

    def __init__(self, level: Union[str, LogLevel] = LogLevel.NONE):
        self.level = LogLevel[level] if isinstance(level, str) else level

    def normal(self, *message):
        print(*message)

    def error(self, *message):
        print('[ERROR]', *message)
        print(format_exc())

    def info(self, *message):
        if self.level >= LogLevel.INFO:
            print('[INFO]', *message)

    def debug(self, *message):
        if self.level >= LogLevel.DEBUG:
            print('[DEBUG]', *message)

    def verbose(self, *message):
        if self.level >= LogLevel.VERBOSE:
            print('[VERBOSE]', *message)
