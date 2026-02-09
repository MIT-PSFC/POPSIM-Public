import fire

from popsim.ml import Trainer, export
from popsim.ml.train_config import TrainConfig
from popsim.modules.meqml.fbt.config import FBT_SURROGATE_CONFIG
from popsim.modules.meqml.fbt.trb import FBTTrainRunBuilder


def export_model(checkpoint_path: str, out_path: str, eval_shots: int = 10):
    """Function to export a trained model checkpoint with the popsim.ml.export function.

    Args:
        checkpoint_path (str): Path to the checkpoint to export.
        out_path (str): Path to save the exported model and validation data.
        eval_shots (int, optional): Maximum number of shots used for validation data. Defaults to 10.
    """
    training_config = TrainConfig(**FBT_SURROGATE_CONFIG)

    train_run_builder = FBTTrainRunBuilder
    _, train_dl, _val_dl, test_dl = train_run_builder.get_dataloaders(training_config.dataloader_config)
    model = train_run_builder.model_init(train_dl, training_config.model_init_config)
    loss_fn = train_run_builder.get_loss_fn(training_config.loss_config)
    opt = train_run_builder.get_optimizer(training_config.optimizer_config)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        checkpoint_dir=training_config.checkpoint_dir,
    )
    trainer.restore_best_checkpoint(checkpoint_path)
    test_dl = test_dl.limit_size(eval_shots, "shot")
    export(trainer.train_state.model, out_path, test_dl)


if __name__ == "__main__":
    fire.Fire(
        {
            "export_model": export_model,
        }
    )
