""" BERT CLASS MODuLE"""

from .embedding import BERTEmbedding
import torch
import torch.nn as nn


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, pad_index, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.pad_index = pad_index

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        self.src_mask = None

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        encoder_layers = nn.TransformerEncoderLayer(hidden, attn_heads, self.feed_forward_hidden, dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def make_src_mask(self, src):
        return src.transpose(0, 1) == self.pad_index
        

    def forward(self, x, segment_info, has_mask=True):
        # self.src_mask = mask
        # if has_mask:
        #     if self.src_mask is None or self.src_mask.size(0) != len(x):
        #         mask = self._generate_square_subsequent_mask(len(x))
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None

        self.src_mask = self.make_src_mask(x)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        x = self.transformer_encoder(x, src_key_padding_mask=self.src_mask)

        return x