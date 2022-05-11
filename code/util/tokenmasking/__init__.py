"""
Package for masking parts of tokenized text.
"""

import warnings
import pkgutil
import importlib

__all__ = []

failed_import = []
for finder, name, is_package in pkgutil.iter_modules(__path__):
    try:
        submodule = importlib.import_module("." + name, __name__)
    except ImportError as ex:
        failed_import.append(name)
        warnings.warn(str(ex), ImportWarning)
        continue
    if hasattr(submodule, "__all__"):
        __all__.extend(submodule.__all__)
        globals().update({element: getattr(submodule, element) for element in submodule.__all__})
if failed_import:
    warnings.warn("Masking algorithms from the following submodules could not be imported: " + ", ".join(failed_import), ImportWarning)

