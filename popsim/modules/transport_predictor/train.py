import fire

from popsim.ml.launch import launch_train
from popsim.modules.transport_predictor.train_configs import BASE_CONFIG


def launch_base():
    launch_train(BASE_CONFIG, use_wandb=True)


def launch_profile_predictor():
    launch_train(BASE_CONFIG.model_init_config["submodules"]["profile_predictor"], use_wandb=True)


if __name__ == "__main__":
    fire.Fire(
        {
            "launch_base": launch_base,
            "launch_profile_predictor": launch_profile_predictor,
        }
    )
