import torch
import torch.nn as nn
import json
import os

# -----------------------------
# CONFIG
# -----------------------------

class Config:

    # Default hyperparameters
    encoder_history = 168
    forecast_length = 168
    encoder_features = 8
    decoder_features = 12
    hidden_size = 512
    num_layers = 1
    dropout = 0.2
    context_dropout = 0.1
    epochs = 1
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
                    print(f"Warning: unknown config key '{key}' in JSON, skipping")

    @classmethod
    def print_config(cls, logger=None):
        logger.info(f"Current config:\n"
        f"                                                             \033[1mbatch size      :\033[0m\033[37m {cls.batch_size}\n"
        f"                                                             \033[1mdecoder features:\033[0m\033[37m {cls.decoder_features}\n"
        f"                                                             \033[1mdevice          :\033[0m\033[37m {cls.device}\n"
        f"                                                             \033[1mdropout         :\033[0m\033[37m {cls.dropout}\n"
        f"                                                             \033[1mencoder features:\033[0m\033[37m {cls.encoder_features}\n"
        f"                                                             \033[1mencoder history :\033[0m\033[37m {cls.encoder_history}\n"
        f"                                                             \033[1mepochs          :\033[0m\033[37m {cls.epochs}\n"
        f"                                                             \033[1mforecast length :\033[0m\033[37m {cls.forecast_length}\n"
        f"                                                             \033[1mhidden size     :\033[0m\033[37m {cls.hidden_size}\n"
        f"                                                             \033[1mlearning rate   :\033[0m\033[37m {cls.learning_rate}\n"
        f"                                                             \033[1mnumber of layers:\033[0m\033[37m {cls.num_layers}\n"
        f"                                                             \033[1moutput size     :\033[0m\033[37m {cls.output_size}\n")


# -----------------------------
# MODEL
# https://arxiv.org/abs/1409.3215: ORIGINAL Paper on "Encoder-Decoder"
# https://arxiv.org/pdf/1409.0473: Videreudvikling, handler om en dårlig første prediction can ødelægge resten.
# -----------------------------
class LSTMForecast(nn.Module):
    def __init__(self, config: Config):
        super(LSTMForecast, self).__init__()
        self.config = config

        self.encoder_lstm = nn.LSTM(
            input_size=config.encoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        self.decoder_lstm = nn.LSTM(
            input_size=config.decoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        self.spread_lstm = nn.LSTM(
            input_size=config.decoder_features,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.spread_init_proj = nn.Linear(config.hidden_size, config.hidden_size // 2)

        self.dropout         = nn.Dropout(config.dropout)
        self.context_dropout = nn.Dropout(config.context_dropout)

        self.fc_q50       = nn.Linear(config.hidden_size,     1)
        self.fc_spread_lo = nn.Linear(config.hidden_size // 2, 1) 
        self.fc_spread_hi = nn.Linear(config.hidden_size // 2, 1) 

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():

            # LSTM input-hidden weights
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)

            # LSTM hidden-hidden weights — orthogonal for gradient stability
            # across 168 timesteps
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)

            # LSTM biases — zeros with forget gate set to 1.0
            elif 'bias' in name and 'fc' not in name and 'proj' not in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias

            # Output projection weights
            elif any(x in name for x in ['fc_q50.weight', 'fc_spread_lo.weight', 'fc_spread_hi.weight']):
                nn.init.xavier_uniform_(param.data)

            # Output projection biases
            elif any(x in name for x in ['fc_q50.bias', 'fc_spread_lo.bias', 'fc_spread_hi.bias']):
                nn.init.constant_(param.data, 0)

            # Spread init projection
            elif 'spread_init_proj.weight' in name:
                nn.init.xavier_uniform_(param.data)

            elif 'spread_init_proj.bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, encoder_input, decoder_input):

        _, (hidden, cell) = self.encoder_lstm(encoder_input)
        hidden = self.context_dropout(hidden)
        cell   = self.context_dropout(cell)

        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        decoder_output    = self.dropout(decoder_output)
        q50               = self.fc_q50(decoder_output).squeeze(-1)

        spread_h0 = torch.tanh(
            self.spread_init_proj(cell[-1])     # [batch, hidden_size // 2]
        ).unsqueeze(0)                          # [1, batch, hidden_size // 2]

        spread_c0 = torch.zeros_like(spread_h0)

        spread_output, _ = self.spread_lstm(decoder_input, (spread_h0, spread_c0))
        spread_output    = self.dropout(spread_output)

        spread_lo = nn.functional.softplus(self.fc_spread_lo(spread_output).squeeze(-1))
        spread_hi = nn.functional.softplus(self.fc_spread_hi(spread_output).squeeze(-1))

        horizon_steps = decoder_input.shape[1]   # 168
        ramp = torch.linspace(0.2, 1.0, horizon_steps, device=decoder_input.device)
        ramp = ramp.unsqueeze(0)                 # [1, 168] — broadcasts over batch

        q10 = q50 - spread_lo * ramp
        q90 = q50 + spread_hi * ramp

        return q10, q50, q90


Config.load_from_file()