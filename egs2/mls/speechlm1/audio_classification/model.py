import torch
import torch.nn as nn


class ESC50Model(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        
        self.embedding = embedding
        self.encoder_config = config.get("encoder_conf")
        self.decoder_config = config.get("decoder_conf")
        self.encoder = nn.TransformerEncoder(**encoder_config)
        
        mlp_module = nn.Sequential(
            nn.Linear(self.encoder_config["d_model"], self.decoder_config["dim_feedforward"]),
            nn.ReLU(),
            nn.Dropout(self.decoder_config["dropout"]),
            nn.Linear(self.decoder_config["dim_feedforward"], self.encoder_config["d_model"]),
        )
        self.decoder = nn.Sequential(
            *[mlp_module for _ in range(self.decoder_config["num_decoder_layers"])]
        )
        
        self.project_dim = config.get("project_dim", 50)
        self.classifier = nn.Linear(eself.decoder_config["d_model"], self.project_dim)

    def forward(self, codec):
        
        codec_embedding = self.embedding(codec)
        x = self.encoder(codec_embedding)
        x = self.decoder(x)
        return self.classifier(x)
    
    