import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForAudioClassification

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
    
    
class PretrainedESC50Model(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()

        self.embedding = embedding
        

        # Define the configuration parameters
        config = AutoConfig.from_pretrained(
            "facebook/wav2vec2-base",  # use a model architecture suitable for audio
            num_labels=50,  # specify the number of classes for classification
            finetuning_task="audio-classification"
        )

        # Initialize the model with the configuration
        model = AutoModelForAudioClassification.from_config(config)
        self.feature_projection = model.wav2vec2.feature_projection
        self.encoder = model.wav2vec2.encoder
        self.projector = model.projector
        self.classifier = model.classifier
        

    def forward(self, codec):

        codec_length = torch.ones(codec.size(0)) * codec.size(1)
        codec_embedding, codec_length = self.embedding(codec, codec_length)
        x = self.feature_projection(codec_embedding)[0]
        x = self.encoder(x)[0]
        x = self.projector(x)
        x = self.classifier(x)
        return x.mean(1)
