import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from mnist.datasets.dataset import MNISTDataModule
from mnist.models.model import LightMNISTClassifier
from mnist.utils.utils import init_trainer


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.training.seed)
    model = LightMNISTClassifier(cfg)

    trainer = init_trainer(cfg)

    data_module = MNISTDataModule(
        val_size=cfg.training.val_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        n1=cfg.preprocessing.n1,
        n2=cfg.preprocessing.n2,
        path_train_val=cfg.data.train_data_path,
        path_test=cfg.data.test_data_path,
    )
    print("Training model...")
    trainer.fit(model=model, datamodule=data_module)
    torch.save(model.state_dict(), cfg.model.model_path + ".pth")
    print(f"Model saved to {cfg.model.model_path}")
    dummy_input_batch = next(iter(data_module.val_dataloader()))[0]
    dummy_input = torch.unsqueeze(dummy_input_batch[0], 0)
    torch.onnx.export(
        model,
        dummy_input,
        cfg.model.model_path + ".onnx",
        export_params=True,
        input_names=["inputs"],
        output_names=["predictions"],
        dynamic_axes={
            "inputs": {0: "BATCH_SIZE"},
            "predictions": {0: "BATCH_SIZE"},
        },
    )


if __name__ == "__main__":
    main()
