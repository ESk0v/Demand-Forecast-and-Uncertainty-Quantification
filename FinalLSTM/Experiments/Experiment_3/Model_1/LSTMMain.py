import os
import torch
from torch.utils.data import DataLoader

from LSTMModel import Config
from LSTMTraining import load_and_split_dataset, train_model


def LSTMMain(filePaths=None, epochs=1, patience=5, logger=None):

    dataset_path = filePaths[0]
    model_save_path = filePaths[1]
# Create checkpoint folder
    run_dir = os.path.dirname(model_save_path)
    os.makedirs(run_dir, exist_ok=True)

    train_dataset, val_dataset, cal_dataset, test_dataset, \
    train_size, val_size, cal_size, test_size = \
        load_and_split_dataset(dataset_path)

    config = Config()
    config.epochs = epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_loader(dataset, shuffle):

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            pin_memory=(device == "cuda"),
            num_workers=min(4, os.cpu_count())
        )

    train_loader = make_loader(train_dataset, True)
    val_loader = make_loader(val_dataset, False)
    cal_loader = make_loader(cal_dataset, False)

    train_model(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        cal_loader=cal_loader,
        train_size=train_size,
        val_size=val_size,
        cal_size=cal_size,
        model_save_path=model_save_path,
        logger=logger,
        patience=patience
    )

    logger.success("Training completed")