"""MetaTT (Lopez-Piqueres et al., TMLR 2026, arXiv:2506.09105): a global tensor-train adapter."""
from .adapter import (MetaTT4D, MetaTT4DConfig, MetaTTLinear, apply_metatt, count_trainable, find_targets,
                      metatt4d_num_params)

__all__ = ["MetaTT4D", "MetaTT4DConfig", "MetaTTLinear", "apply_metatt", "count_trainable", "find_targets",
           "metatt4d_num_params"]
