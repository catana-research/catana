import logging
import time
from functools import wraps


class Timer(object):
    """ Timer

    Timer context manager.

    Examples
    --------

    with Timer() as t:
        x = 0
        for i in range(1000):
            x += 1
    t.elapsed()

    """
    def __init__(self, log_name=__name__, log_level=logging.INFO, verbose=True, precision=4):
        self.log_name = log_name
        self.log_level = log_level
        self.verbose = verbose
        self.precision = precision

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self._elapsed = self.end - self.start
        message = f'Time elapsed = {self._elapsed:.{self.precision}f}'
        self._log(message)
        if self.verbose:
            print(message)

    def _log(self, message):
        logger = logging.getLogger(self.log_name)
        logger.log(self.log_level, message)

    def __call__(self, f):
        """Enables Timer to be used as a decorator.

        Examples
        --------
        @Timer('Test')
        def x():
            pass

        """
        @wraps(f)
        def _wrapper(*args, **kwargs):
            with self:
                return f(*args, **kwargs)

        return _wrapper

    def elapsed(self):
        return self._elapsed
