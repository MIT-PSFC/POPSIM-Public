import fire

from popsim.ml.launch import launch_train
from popsim.modules.profile_predictor.train_configs import DUMMY_CONFIG, TCV_CONFIG


def launch_dummy():
    launch_train(DUMMY_CONFIG, use_wandb=True)


def launch_tcv():
    launch_train(TCV_CONFIG, use_wandb=True)


if __name__ == "__main__":
    fire.Fire(
        {
            "launch_dummy": launch_dummy,
            "launch_tcv": launch_tcv,
        }
    )
