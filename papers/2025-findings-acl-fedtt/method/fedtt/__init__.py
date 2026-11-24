"""FedTT / FedTT+ (Ghiasvand et al., Findings of ACL 2025): the method proposed in the paper.

  tt.py         tensor-train linear layer (Eq. 1) and the FedTT+ core schedule (Alg. 2, line 3)
  model.py      tensorized adapters and tensorized classifier inserted into a Hugging Face encoder,
                FedTT+ freezing per round (and LoRA-family references for side-by-side checks)
  algorithm.py  one communication round of Algorithm 1 (FedTT) / Algorithm 2 (FedTT+): local AdamW updates
                on each selected client and server-side averaging of exactly the uploaded parameters

The experiment code (GLUE data, client splits, suites, Colab notebooks) is in ../../reproduction.
"""
from .algorithm import local_batches, run_round, select_clients
from .model import FedModel, TensorizedAdapter, build_model
from .tt import TTLinear, plus_core_index, split_modes

__version__ = "0.1.0"
