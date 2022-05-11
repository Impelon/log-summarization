from ..__init__ import CacheWithEnv, SwitchableCacheAggregate

import pkgutil
import importlib
import os.path

__all__ = ["internal_cache", "cache"]

internal_cache = CacheWithEnv(os.path.dirname(os.path.realpath(__file__)), "HF_HOME")

submodule_caches = {}
for finder, name, is_package in pkgutil.iter_modules(__path__):
    submodule = importlib.import_module("." + name, __name__)
    if hasattr(submodule, "cache"):
        submodule_caches[name] = submodule.cache

cache = SwitchableCacheAggregate(internal_cache=internal_cache, **submodule_caches)
