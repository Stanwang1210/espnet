#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding Frontend for text based inputs."""

from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Embedding(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    @typechecked
    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        # TODO(sdalmia): check for padding idx
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(input_size, embed_dim),
            pos_enc_class(embed_dim, positional_dropout_rate),
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T) or (B, T,D), with D.
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D).
            Tensor: Output lengths within batch.
        """
        x = self.embed(input)

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim


class PatchEmbedding(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    @typechecked
    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        patch_size: int = 1,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            patch_size: number of token per patch to sum up the embeddings
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.emb = torch.nn.Embedding(input_size, embed_dim)
        self.pos = pos_enc_class(embed_dim, positional_dropout_rate)
        self.ln = torch.nn.LayerNorm(embed_dim)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T)
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T // patch_size, D).
            Tensor: Output lengths within batch, devided by patch_size
        """

        assert input.dim() == 2, input.size()
        assert input.size(1) % self.patch_size == 0, input.size()
        assert torch.all(input_lengths % self.patch_size == 0), input_lengths

        B, T = input.size()
        x = input.view(B, T // self.patch_size, self.patch_size)
        x = self.emb(x).mean(dim=2)
        x = self.ln(self.pos(x))

        input_lengths = input_lengths // self.patch_size

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim


class CodecEmbedding(AbsFrontend):
    """Use codec dequantization process and the input embeddings"""

    @typechecked
    def __init__(
        self,
        input_size,
        token_bias: int = 2,
        patch_size: int = 8,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
        codec_conf: dict = {
            "codec_choice": "ESPnet",
            "codec_fs": 16000,
            "hf_model_tag": "espnet/amuse_encodec_16k",
        },
    ):
        """Initialize.

        Args:
            hf_model_tag: HuggingFace model tag for Espnet codec models
            token_bias: the index of the first codec code
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """

        super().__init__()

        from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer

        model = CodecTokenizer(**codec_conf)
        
        self.quantizer = model.get_quantizer()
        self.token_bias = token_bias
        self.codebook_size = model.n_codebook
        self.codebook_dim = model.size_codebook

        # NOTE(Jinchuan): make it as an external parameter rather than parsing from
        # the quantizer since not all codebooks will be used all the time.
        self.n_codebook = patch_size

        self.vocab_size = input_size
        self.pos = pos_enc_class(self.codebook_dim, positional_dropout_rate)
        self.ln = torch.nn.LayerNorm(self.codebook_dim)

        # self.decoder = model.codec.generator.decoder

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        assert input.dim() == 2, input.size()
        assert input.size(1) % self.n_codebook == 0, input.size()
        assert torch.all(input_lengths % self.n_codebook == 0), (
            input_lengths,
            input_lengths % self.n_codebook,
        )
        assert torch.all(input < self.vocab_size)

        B, Tnq = input.size()
        x = input.view(B, Tnq // self.n_codebook, self.n_codebook)
        x = x - self.token_bias

        for n in range(self.n_codebook):
            x[:, :, n] -= n * self.codebook_size
        # NOTE (Jinchuan): do this clip so that the dequantization process
        # will not encounter an error. In practice, only the padding values
        # will exceed this range and is ignored due to the length masking.
        x = torch.clip(x, min=0, max=self.codebook_size - 1)

        z = self.quantizer.decode(x.permute(2, 0, 1)).permute(0, 2, 1)

        z = self.ln(z)
        z = self.pos(z)

        input_lengths = input_lengths // self.n_codebook

        return z, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.codebook_dim


if __name__ == "__main__":
    import torch
    import yaml
    import warnings

    warnings.filterwarnings("ignore")

    from espnet2.mt.frontend.embedding import CodecEmbedding

    # test Codec Embedding
    config_paths = [
        "conf/tuning/train_asr_ebranchformer_DAC.yaml",
        "conf/tuning/train_asr_ebranchformer_EnCodec.yaml",
        "conf/tuning/train_asr_ebranchformer_ESPnet.yaml",
    ]
    for config_path in config_paths:
        print(
            "----------------------------------------------------------------------------------------"
        )
        print(f"Config: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        try:
            frontend = CodecEmbedding(input_size=16, **config["frontend_conf"])
        except Exception as e:
            if isinstance(e, NotImplementedError):
                print(f"Fail!! Config {config_path} fails to passed.")
                continue
            else:
                raise e

        print(f"Quantizer: {frontend.quantizer}")
        print(f"Token Bias: {frontend.token_bias}")
        print(f"Codebook Size: {frontend.codebook_size}")
        print(f"Codebook Dimension: {frontend.codebook_dim}")
        print(f"Number of Codebook: {frontend.n_codebook}")
        print(f"Vocabulary Size: {frontend.vocab_size}")

        print(f"Success !! Config {config_path} passed.")
        del frontend
        print(
            "--------------------------------------------------------------------------------------------"
        )
