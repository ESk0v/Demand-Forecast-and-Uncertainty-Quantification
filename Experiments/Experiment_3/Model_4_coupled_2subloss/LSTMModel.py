import torch
import torch.nn as nn
import json
import os

# =========================================================
# CONFIG
# =========================================================

class Config:

    encoder_history = 168
    forecast_length = 168

    encoder_features = 8
    decoder_features = 12

    hidden_size = 512
    num_layers = 1

    dropout = 0.2
    context_dropout = 0.1

    epochs = 50
    batch_size = 64
    learning_rate = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_size = 1

    tuned_config_path = "NewModelFolder/Files/HPTTuning.json"

    @classmethod
    def load_from_file(cls):
        path = cls.tuned_config_path
        if os.path.exists(path):
            with open(path, "r") as f:
                params = json.load(f)

            for key, value in params.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
                else:
                    print(f"[WARN] Unknown config key: {key}")

    @classmethod
    def print_config(cls, logger=None):
        msg = (
            f"batch_size: {cls.batch_size}\n"
            f"decoder_features: {cls.decoder_features}\n"
            f"device: {cls.device}\n"
            f"dropout: {cls.dropout}\n"
            f"encoder_features: {cls.encoder_features}\n"
            f"encoder_history: {cls.encoder_history}\n"
            f"epochs: {cls.epochs}\n"
            f"forecast_length: {cls.forecast_length}\n"
            f"hidden_size: {cls.hidden_size}\n"
            f"learning_rate: {cls.learning_rate}\n"
            f"num_layers: {cls.num_layers}\n"
            f"output_size: {cls.output_size}\n"
        )

        if logger:
            logger.info(msg)
        else:
            print(msg)


# =========================================================
# MODEL
# =========================================================

class LSTMForecast(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # -------------------------
        # Encoder
        # -------------------------
        self.encoderLstm = nn.LSTM(
            input_size=config.encoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        # -------------------------
        # Decoder (shared trunk)
        # -------------------------
        self.decoderLstm = nn.LSTM(
            input_size=config.decoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(config.dropout)
        self.context_dropout = nn.Dropout(config.context_dropout)

        # -------------------------
        # 3 quantile heads
        # -------------------------
        self.fc_q10 = nn.Linear(config.hidden_size, 1)
        self.fc_q50 = nn.Linear(config.hidden_size, 1)
        self.fc_q90 = nn.Linear(config.hidden_size, 1)

        self._init_weights()

    # =====================================================
    # WEIGHT INITIALISATION
    # =====================================================
    def _init_weights(self):
        for name, param in self.named_parameters():

            # LSTM input weights
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)

            # LSTM recurrent weights
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)

            # LSTM biases
            elif "bias" in name and "fc" not in name:
                nn.init.constant_(param.data, 0)

                # forget gate bias trick (stability)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

            # Fully connected layers
            elif "fc_" in name and "weight" in name:
                nn.init.xavier_uniform_(param.data)

            elif "fc_" in name and "bias" in name:
                nn.init.constant_(param.data, 0)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, encoder_input, decoder_input):

        _, (hidden, cell) = self.encoderLstm(encoder_input)

        hidden = self.context_dropout(hidden)
        cell   = self.context_dropout(cell)

        decoder_output, _ = self.decoderLstm(decoder_input, (hidden, cell))
        decoder_output = self.dropout(decoder_output)

        q10 = self.fc_q10(decoder_output).squeeze(-1)
        q50 = self.fc_q50(decoder_output).squeeze(-1)
        q90 = self.fc_q90(decoder_output).squeeze(-1)

        return q10, q50, q90