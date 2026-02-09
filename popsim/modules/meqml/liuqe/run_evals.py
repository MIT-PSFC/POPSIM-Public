import fire

from popsim.ml.checkpointing import restore_model_from_path
from popsim.ml.eval import run_evals
from popsim.modules.meqml.liuqe import LATEST_CHECKPOINT
from popsim.modules.meqml.liuqe.config import LIUQE_CONFIG
from popsim.modules.meqml.liuqe.evals import eval_padded_relative_error_medians
from popsim.modules.meqml.liuqe.trb import LIUQETrainRunBuilder


def restore_latest():
    _, train_dl, val_dl, test_dl = LIUQETrainRunBuilder.get_dataloaders(LIUQE_CONFIG["dataloader_config"])
    module = LIUQETrainRunBuilder.model_init(train_dl, LIUQE_CONFIG["model_init_config"])

    module = restore_model_from_path(LATEST_CHECKPOINT, module)
    return module, train_dl, val_dl, test_dl


def run_simple_evals():
    module, _, _, test_dl = restore_latest()
    eval_suite = {"padded_relative_error_medians": eval_padded_relative_error_medians}

    results = run_evals(module, test_dl, eval_suite)
    print(results["padded_relative_error_medians"])


if __name__ == "__main__":
    fire.Fire()
