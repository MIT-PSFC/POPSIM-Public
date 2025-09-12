import fire

from popsim.ml.launch import launch_train
from popsim.modules.profile_predictor.train_configs import SPARC_CONFIG, TCV_CONFIG


def launch_sparc():
    launch_train(SPARC_CONFIG, use_wandb=True)


def launch_tcv():
    launch_train(TCV_CONFIG, use_wandb=True)


if __name__ == "__main__":
    fire.Fire(
        {
            "launch_sparc": launch_sparc,
            "launch_tcv": launch_tcv,
        }
    )
