import torch
import torch.nn as nn


class ESC50Model(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        
        self.embedding = embedding
        self.encoder_config = config.get(
            "encoder_conf", 
            {"d_model": 256, "nhead": 4, "num_encoder_layers": 2, "dim_feedforward": 512, "dropout": 0.1}
        )
        self.encoder = nn.TransformerEncoder(**encoder_config)
        
        self.decoder_config = config.get(
            "decoder_conf", 
            {"num_decoder_layers": 2, "dim_feedforward": 512, "dropout": 0.1}
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
        self.classifier = nn.Linear(encoder_config["d_model"], project_dim)

    def forward(self, codec):
        
        codec_embedding = self.embedding(codec)
        x = self.encoder(codec_embedding)
        x = self.decoder(x)
        return self.classifier(x)
    
    