from torch.utils.data import DataLoader
from LSTMModel import Config
import os
import torch

from LSTMTraining import load_and_split_dataset, train_model

CONFORMAL_ALPHA = 0.10

def LSTMMain(filePaths=None, epochs=1, patience=5, logger=None):
    dataset_path    = filePaths[0]
    model_save_path = filePaths[1]
    run_dir = os.path.dirname(model_save_path)
    os.makedirs(run_dir, exist_ok=True)

    train_dataset, val_dataset, cal_dataset, test_dataset, \
        train_size, val_size, cal_size, test_size = load_and_split_dataset(dataset_path)

    unique_total = train_size + val_size + test_size  # cal overlaps val in this split.
    logger.info(
        f"Dataset loaded: {train_size} train  {val_size} val  "
        f"{cal_size} cal(overlap)  {test_size} test  (unique total: {unique_total})"
    )

    config = Config()
    config.epochs = epochs

    num_workers = min(4, os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_loader(dataset, shuffle, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(device == 'cuda'),
            num_workers=num_workers,
            persistent_workers=False,
        )

    train_loader = make_loader(train_dataset, shuffle=True,  batch_size=config.batch_size)
    val_loader   = make_loader(val_dataset,   shuffle=False, batch_size=config.batch_size * 8)
    cal_loader   = make_loader(cal_dataset,   shuffle=False, batch_size=config.batch_size * 8)
    test_loader  = make_loader(test_dataset,  shuffle=False, batch_size=config.batch_size * 8)

    if device == "cuda":
        torch.cuda.empty_cache()

    best_val_loss, train_losses, val_losses = train_model(
        config, train_loader, val_loader, cal_loader, test_loader,
        train_size, val_size, cal_size, test_size,
        model_save_path, dataset_path,
        logger=logger,
        patience        = patience,
        conformal_alpha = CONFORMAL_ALPHA,
        model_name="Model_5 Coupled - 1 loss",
        training_variant="coupled",
    )

    checkpoint = torch.load(model_save_path, weights_only=False)
    logger.success("LSTM training completed successfully!")
    logger.info("Training plots generated in run checkpoint folder.")

    #generate_training_readme(
    #    plot_dir       = run_dir,
    #    model_filename = os.path.basename(model_save_path),
    #    config         = config,
    #    train_size     = train_size,
    #    val_size       = val_size,
    #    cal_size       = cal_size,
    #    test_size      = test_size,
    #    n_total        = n_total,
    #    epochs_run     = len(train_losses),
    #    best_epoch     = checkpoint['epoch'],
    #    best_val_loss  = checkpoint['val_loss'],
    #    conformal_u_alpha = checkpoint['conformal_u_alpha'],
    #    early_stopped  = (len(train_losses) < config.epochs),
    #    patience       = patience
    #)

    #logger.success("Training README successfully generated")
