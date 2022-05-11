from ..__init__ import CacheWithEnv

import os.path

__all__ = ["cache"]

cache = CacheWithEnv(os.path.dirname(os.path.realpath(__file__)), "NLTK_DATA")
