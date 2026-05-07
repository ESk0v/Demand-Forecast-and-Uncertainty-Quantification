class LSTMForecast(nn.Module):
    def __init__(self, config):
        super(LSTMForecast, self).__init__()
        self.config = config

        # ─────────────────────────────
        # ENCODER
        # ─────────────────────────────
        self.encoderLstm = nn.LSTM(
            input_size=config.encoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        # ─────────────────────────────
        # SINGLE DECODER (shared trunk)
        # ─────────────────────────────
        self.decoderLstm = nn.LSTM(
            input_size=config.decoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(config.dropout)
        self.context_dropout = nn.Dropout(config.context_dropout)

        # ─────────────────────────────
        # THREE OUTPUT HEADS
        # ─────────────────────────────
        self.fc_q10 = nn.Linear(config.hidden_size, 1)
        self.fc_q50 = nn.Linear(config.hidden_size, 1)
        self.fc_q90 = nn.Linear(config.hidden_size, 1)

        self._init_weights()

    # ─────────────────────────────
    # INITIALISATION (simplified but stable)
    # ─────────────────────────────
    def _init_weights(self):
        for name, param in self.named_parameters():

            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)

            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)

            elif 'bias' in name and 'fc' not in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

            elif 'fc_' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)

            elif 'fc_' in name and 'bias' in name:
                nn.init.constant_(param.data, 0)

    # ─────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────
    def forward(self, encoder_input, decoder_input):

        # Encoder
        _, (hidden, cell) = self.encoderLstm(encoder_input)

        hidden = self.context_dropout(hidden)
        cell   = self.context_dropout(cell)

        # Decoder
        decoder_output, _ = self.decoderLstm(
            decoder_input,
            (hidden, cell)
        )

        decoder_output = self.dropout(decoder_output)

        # ─────────────────────────────
        # DIRECT 3-HEAD OUTPUT
        # ─────────────────────────────
        q10 = self.fc_q10(decoder_output).squeeze(-1)
        q50 = self.fc_q50(decoder_output).squeeze(-1)
        q90 = self.fc_q90(decoder_output).squeeze(-1)

        return q10, q50, q90