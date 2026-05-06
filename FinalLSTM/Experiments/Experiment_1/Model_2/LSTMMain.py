from torch.utils.data import DataLoader
from LSTMModel import Config
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTM.LSTMTraining import load_and_split_dataset, train_model
from LSTM.GenerateREADME import generate_training_readme

CONFORMAL_ALPHA = 0.60

def LSTMMain(filePaths=None, epochs=1, patience=5, logger=None):
    dataset_path    = filePaths[0]
    model_save_path = filePaths[1]
    run_dir = os.path.dirname(model_save_path)
    os.makedirs(run_dir, exist_ok=True)

    train_dataset, val_dataset, cal_dataset, test_dataset, \
        train_size, val_size, cal_size, test_size = load_and_split_dataset(dataset_path)

    n_total = train_size + val_size + cal_size + test_size
    logger.info(
        f"Dataset loaded: {train_size} train  {val_size} val  "
        f"{cal_size} cal  {test_size} test  (total: {n_total})"
    )

    config = Config()
    config.epochs = epochs

    num_workers = min(4, os.cpu_count())
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

    if device == "cuda":
        torch.cuda.empty_cache()

    best_val_loss, train_losses, val_losses = train_model(
        config, train_loader, val_loader, cal_loader,
        train_size, val_size, cal_size,
        model_save_path, logger,
        patience        = patience,
        conformal_alpha = CONFORMAL_ALPHA,
    )

    checkpoint = torch.load(model_save_path, weights_only=False)
    logger.success("LSTM training completed successfully!")
    logger.info("Generating training README...")

    logger.success("Training README successfully generated")

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