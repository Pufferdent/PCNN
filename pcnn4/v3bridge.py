"""Make PC-NN-v3's pipeline modules importable.

v4 reuses v3's battle-tested machinery — sfinder invocation, fumen
glue/order enumeration, V* lookup, stream reconstruction — instead of
forking it. v3's modules import each other as top-level names (``config``,
``data_pipeline``, ...), so its pc-nn directory goes on sys.path (appended,
so nothing in v4 can be shadowed); v3's own ``_sdk_bootstrap`` then pulls in
the Tetris SDK.
"""
import os
import sys

from pcnn4.config import PATHS

_dir = PATHS['v3_pcnn_dir']
if not os.path.isdir(_dir):
    raise ImportError(
        "PC-NN-v3 pc-nn directory not found at %r; set PCNN_V3_PATH" % _dir)
if _dir not in sys.path:
    sys.path.append(_dir)
