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
    hidden_size = 768
    num_layers = 2
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
        # Print all class attributes that are hyperparameters
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

        # ===== ENCODER LSTM ======
        # The encoder's job is to READ and COMPRESS the input sequence (your
        # historical/context window) into a fixed-size summary called the
        # "context vector" or "thought vector".
        #
        # Mechanically, it processes each timestep t=1..T of the input sequence
        # and updates its internal state (h, c) at every step:
        #
        #   h_t, c_t = LSTM_cell(x_t, h_{t-1}, c_{t-1})
        #
        # where:
        #   x_t  → input at timestep t    (shape: [batch, encoder_features])
        #   h_t  → hidden state           (shape: [num_layers, batch, hidden_size])
        #   c_t  → cell state             (shape: [num_layers, batch, hidden_size])
        #
        # After processing the FULL input sequence, the FINAL (h_T, c_T) pair
        # acts as the compressed memory of everything the encoder saw.
        # This is what gets handed off to the decoder.
        self.encoder_lstm = nn.LSTM(
            input_size=config.encoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        # ====== DECODER LSTM =======
        # The decoder's job is to GENERATE the output sequence (your forecast
        # horizon) step-by-step, conditioned on the encoder's final state.
        #
        # It is initialized with the encoder's (h_T, c_T), meaning it "starts"
        # with the compressed memory of the input sequence already loaded.
        # It then autoregressively produces outputs one timestep at a time:
        #
        #   ŷ_t, (h_t, c_t) = LSTM_cell(d_t, h_{t-1}, c_{t-1})
        self.decoder_lstm = nn.LSTM(
            input_size=config.decoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(config.dropout)  # ← add this line here
        self.context_dropout = nn.Dropout(config.context_dropout)  # ← add this line here

        # ─── OUTPUT PROJECTION ──────────────────────────────────────────────────
        # Projects the decoder's hidden state at each timestep from hidden_size
        # dimensions down to output_size (e.g. 1 for univariate point forecast,
        # or N for multi-target / quantile forecasting).
        #
        #   ŷ_t = W · h_t + b     where W ∈ ℝ^{output_size × hidden_size}
        
        self.fc_q10 = nn.Linear(config.hidden_size, 1)
        self.fc_q50 = nn.Linear(config.hidden_size, 1)
        self.fc_q90 = nn.Linear(config.hidden_size, 1)

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
            elif 'bias' in name and 'fc' not in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias

            # Output projection weights
            elif any(x in name for x in ['fc_q10.weight', 'fc_q50.weight', 'fc_q90.weight']):
                nn.init.xavier_uniform_(param.data)

            # Output projection biases
            elif any(x in name for x in ['fc_q10.bias', 'fc_q50.bias', 'fc_q90.bias']):
                nn.init.constant_(param.data, 0)

    def forward(self, encoder_input, decoder_input):
        _, (hidden, cell) = self.encoder_lstm(encoder_input)

        hidden = self.context_dropout(hidden)
        cell   = self.context_dropout(cell)

        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        decoder_output = self.dropout(decoder_output)

        q10 = self.fc_q10(decoder_output).squeeze(-1)
        q50 = self.fc_q50(decoder_output).squeeze(-1)
        q90 = self.fc_q90(decoder_output).squeeze(-1)

        return q10, q50, q90
    
Config.load_from_file()