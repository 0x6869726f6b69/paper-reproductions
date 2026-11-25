"""Reproduction of MetaTT (arXiv:2506.09105) Table 2, RoBERTa-base, MetaTT-4D rank 8 on GLUE MRPC and RTE."""
import os
import sys

try:                                   # the method lives in ../../method/metatt in the repository layout
    import metatt  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "method"))
