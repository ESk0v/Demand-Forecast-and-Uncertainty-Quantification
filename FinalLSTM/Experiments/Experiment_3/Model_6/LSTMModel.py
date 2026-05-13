import torch
import torch.nn as nn
import json
import os

# -----------------------------
# CONFIG
# -----------------------------

class Config:

    # Default hyperparameters
    encoder_history  = 168
    forecast_length  = 168
    encoder_features = 8
    decoder_features = 12
    hidden_size      = 512
    num_layers       = 1
    dropout          = 0.2
    context_dropout  = 0.1
    epochs           = 1
    batch_size       = 64
    learning_rate    = 1e-4
    device           = "cuda" if torch.cuda.is_available() else "cpu"
    output_size      = 1

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
        logger.info(
            f"Current config:\n"
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
            f"                                                             \033[1moutput size     :\033[0m\033[37m {cls.output_size}\n"
        )


# -----------------------------
# MODEL
# https://arxiv.org/abs/1409.3215 : Original Encoder-Decoder paper
# https://arxiv.org/pdf/1409.0473 : Attention extension — early prediction errors
# -----------------------------

class LSTMForecast(nn.Module):
    """
    Encoder-decoder LSTM with a separate uncertainty head.

    Sequential training phases
    ──────────────────────────
    Phase 1 — median:
        Only the encoder, decoder, and fc_decoderMedian are trained.
        The uncertainty head receives no gradients and its parameters
        stay at their initialised values.

    Phase 2 — spread:
        The encoder and decoder are frozen (weights fixed from phase 1).
        Only the uncertainty LSTM and its projection heads are trained,
        giving the spread a stable, converged q50 target to model against.

    Inference (training_phase=None, default):
        All sub-networks are active; frozen/unfrozen state follows whatever
        freeze_for_phase() last set, or the default (all trainable).
    """

    def __init__(self, config: Config):
        super(LSTMForecast, self).__init__()
        self.config = config

        self.encoderLstm = nn.LSTM(
            input_size  = config.encoder_features,
            hidden_size = config.hidden_size,
            num_layers  = config.num_layers,
            batch_first = True,
            dropout     = config.dropout if config.num_layers > 1 else 0.0,
        )

        self.decoderLstm = nn.LSTM(
            input_size  = config.decoder_features,
            hidden_size = config.hidden_size,
            num_layers  = config.num_layers,
            batch_first = True,
            dropout     = config.dropout if config.num_layers > 1 else 0.0,
        )

        self.uncertaintyDecoderLstm = nn.LSTM(
            input_size  = config.decoder_features,
            hidden_size = config.hidden_size,
            num_layers  = 1,
            batch_first = True,
            dropout     = 0.0,
        )

        self.dropout                  = nn.Dropout(config.dropout)
        self.context_dropout          = nn.Dropout(config.context_dropout)
        self.ramp_layer               = nn.Linear(1, 1)
        self.fc_decoderMedian         = nn.Linear(config.hidden_size, 1)
        self.fc_uncertaintyDecoderLow = nn.Linear(config.hidden_size, 1)
        self.fc_uncertaintyDecoderHigh= nn.Linear(config.hidden_size, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        for name, param in self.named_parameters():

            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)

            elif 'weight_hh' in name:
                # Orthogonal init for gradient stability over 168 timesteps
                nn.init.orthogonal_(param.data)

            elif 'bias' in name and 'fc' not in name and 'proj' not in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)   # forget-gate bias = 1

            elif any(x in name for x in [
                'fc_decoderMedian.weight',
                'fc_uncertaintyDecoderLow.weight',
                'fc_uncertaintyDecoderHigh.weight',
            ]):
                nn.init.xavier_uniform_(param.data)

            elif any(x in name for x in [
                'fc_decoderMedian.bias',
                'fc_uncertaintyDecoderLow.bias',
                'fc_uncertaintyDecoderHigh.bias',
            ]):
                nn.init.constant_(param.data, 0)

            elif 'ramp_layer.weight' in name:
                nn.init.constant_(param.data, 2.0)
            elif 'ramp_layer.bias' in name:
                nn.init.constant_(param.data, -1.0)

    # ------------------------------------------------------------------
    # Phase helpers  (called by the training loop, not forward)
    # ------------------------------------------------------------------
    def freeze_for_phase(self, phase: int):
        median_modules = [
            self.encoderLstm,
            self.decoderLstm,
            self.fc_decoderMedian,
            self.context_dropout,
        ]
        spread_modules = [
            self.uncertaintyDecoderLstm,
            self.fc_uncertaintyDecoderLow,
            self.fc_uncertaintyDecoderHigh,
            self.ramp_layer,
        ]

        if phase == 1:
            for m in median_modules:
                for p in m.parameters():
                    p.requires_grad = True
            for m in spread_modules:
                for p in m.parameters():
                    p.requires_grad = False

        elif phase == 2:
            for m in median_modules:
                for p in m.parameters():
                    p.requires_grad = False
            for m in spread_modules:
                for p in m.parameters():
                    p.requires_grad = True

        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, encoder_input, decoder_input):
        _, (hidden, cell) = self.encoderLstm(encoder_input)
        hidden = self.context_dropout(hidden)
        cell   = self.context_dropout(cell)

        decoder_output, _ = self.decoderLstm(decoder_input, (hidden, cell))
        decoder_output    = self.dropout(decoder_output)
        q50               = self.fc_decoderMedian(decoder_output).squeeze(-1)

        spread_output, _ = self.uncertaintyDecoderLstm(
            decoder_input, (hidden.detach(), cell.detach())
        )
        spread_output = self.dropout(spread_output)

        spread_lo = nn.functional.softplus(
            self.fc_uncertaintyDecoderLow(spread_output).squeeze(-1)
        )
        spread_hi = nn.functional.softplus(
            self.fc_uncertaintyDecoderHigh(spread_output).squeeze(-1)
        )

        horizon_steps = decoder_input.shape[1]
        ramp = torch.sigmoid(self.ramp_layer(torch.linspace(0, 1, horizon_steps, device=decoder_input.device).unsqueeze(-1))).squeeze(-1).unsqueeze(0)

        q10 = q50 - spread_lo * ramp
        q90 = q50 + spread_hi * ramp

        return q10, q50, q90