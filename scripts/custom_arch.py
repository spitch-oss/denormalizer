#!/usr/bin/env python3

import re
import sys

from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture
from fairseq_cli.train import cli_main

@register_model_architecture("transformer", "transformer_normalizer")
def transformer_normalizer(args) :    
    # Embeddings and layer normalization location
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    # Dropout
    args.dropout = getattr(args, "dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    return base_architecture(args)


@register_model_architecture("transformer", "transformer_small")
def small_architecture(args):
    # Embeddings and layer normalization location
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    # Dropout
    args.dropout = getattr(args, "dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)

    # Encoder
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    # Decoder
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    return base_architecture(args)


if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cli_main())
