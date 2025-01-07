import popsim.ml.xarray_accessor  # noqa: F401
from popsim.ml.dataloading import DataLoader, make_dataloader, make_time_indep_dataloader
from popsim.ml.eval import EvalData
from popsim.ml.loss import IntegralLoss
from popsim.ml.split_utils import split_dataset_along_dim
from popsim.ml.trainer import Trainer

DEFAULT_SAMPLE_DIM = "sample"

__all__ = [
    "split_dataset_along_dim",
    "make_dataloader",
    "make_time_indep_dataloader",
    "Trainer",
    "IntegralLoss",
    "EvalData",
    "DataLoader",
    "DEFAULT_SAMPLE_DIM",
]
