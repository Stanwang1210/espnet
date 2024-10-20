import torch
import torch.nn as nn


class ESC50Model(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()

        self.embedding = embedding
        encoder_config = config.get("encoder_conf")
        decoder_config = config.get("decoder_conf")
        num_layers = encoder_config.pop("num_encoder_layers")
        self.pre_encoder = nn.Linear(
            self.embedding.codebook_dim, encoder_config["d_model"]
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(batch_first=True, **encoder_config),
            num_layers=num_layers,
        )

        mlp_module = nn.Sequential(
            nn.Linear(encoder_config["d_model"], decoder_config["dim_feedforward"]),
            nn.ReLU(),
            nn.Dropout(decoder_config["dropout"]),
            nn.Linear(decoder_config["dim_feedforward"], encoder_config["d_model"]),
        )
        self.decoder = nn.Sequential(
            *[mlp_module for _ in range(decoder_config["num_decoder_layers"])]
        )

        self.project_dim = config.get("project_dim", 50)
        self.classifier = nn.Linear(encoder_config["d_model"], self.project_dim)

    def forward(self, codec):

        codec_length = torch.ones(codec.size(0)) * codec.size(1)
        codec_embedding, codec_length = self.embedding(codec, codec_length)
        x = self.pre_encoder(codec_embedding)
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.decoder(x)
        return self.classifier(x)
