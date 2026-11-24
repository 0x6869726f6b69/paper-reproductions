"""Reproduction experiments for FedTT / FedTT+ (Ghiasvand et al., Findings of ACL 2025).

The method itself is the package `fedtt` in ../../method (on Colab both packages sit side by side in
/content/fedtt_code). This makes `import fedtt` work when running from the reproduction folder.
"""
import os
import sys

_METHOD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "method")
if os.path.isdir(os.path.join(_METHOD, "fedtt")) and _METHOD not in sys.path:
    sys.path.insert(0, _METHOD)

__version__ = "0.1.0"
