#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torchtext
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from nltk.translate.bleu_score import corpus_bleu 
import io
import time
import pandas as pd
import numpy as np
import pickle
import sentencepiece as spm
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from bs4 import BeautifulSoup, SoupStrainer


torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.get_device_name(0))


hi_model_path = 'sentencepiece/gu_hi/tokenizer_32000'
gu_tokenizer = spm.SentencePieceProcessor(model_file=f'{hi_model_path}.model')


with open('sentencepiece/gu_hi/tokenizer_32000.vocab', encoding='utf-8') as f:
    gu_vocab = [doc.strip().split("\t") for doc in f]

gu_word2idx = {w[0]: i for i, w in enumerate(gu_vocab)}

with open('sentencepiece/en_gu_hi/tokenizer_32000.vocab', encoding='utf-8') as f:
    en_vocab = [doc.strip().split("\t") for doc in f]

en_word2idx = {w[0]: i for i, w in enumerate(en_vocab)}

BATCH_SIZE = 64 
PAD_IDX = gu_word2idx['[PAD]']
BOS_IDX = gu_word2idx['[BOS]']
EOS_IDX = gu_word2idx['[EOS]']


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.3):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


SRC_VOCAB_SIZE = 32000
TGT_VOCAB_SIZE = 32000
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_EPOCHS = 16

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)



model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)

checkpoint = torch.load('model_checkpoint1.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    tokens = [BOS_IDX] + src_tokenizer.encode(src) + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    sent = []
    for tok in tgt_tokens:
        for word, idx in tgt_vocab.items():
            if idx == tok:
                sent.append(word)
    sent = "".join(sent).replace("[BOS]", "").replace("[EOS]", "").replace("▁", " ").lstrip()
    
    return sent


target = []

with open('datasets/newstest2019-guen-ref.en/newstest2019-guen-ref.en.sgm', 'r') as f:
    data = f.read()
    soup = BeautifulSoup(data, 'lxml')
    seg_attrs = soup.findAll('seg')
        
for i, attr in enumerate(seg_attrs):
    target.append(attr.text)


source = []
with open('datasets/newstest2019-guen-ref.en/newstest2019-guen-src.gu.sgm', 'r') as f:
    data = f.read()
    soup = BeautifulSoup(data, 'lxml')
    seg_attrs = soup.findAll('seg')
        
for i, attr in enumerate(seg_attrs):
    source.append(attr.text)


def transliterate(sent):
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
    
    source = UnicodeIndicTransliterator.transliterate(sent, 'gu', 'hi')
    
    return source


def bleu_score(source, target, model, src_vocab, tgt_vocab, src_tokenizer):
    
    import string
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    
    score = []
    
    for src, tgt in zip(source, target):
        
    
        #for excluding puctuation on English
        table = str.maketrans(dict.fromkeys(string.punctuation))

        # transliterate gujarati 
        src = transliterate(src)
        ref = tgt.translate(table)
    
        candidate = translate(model, src, src_vocab, tgt_vocab, src_tokenizer).split()
        ref_splited = ref.split()
    
        score.append(sentence_bleu(ref_splited, candidate, weights=(1, 0, 0, 0)))
    
    return score
    

bleu = bleu_score(source, target, model, gu_word2idx, en_word2idx, gu_tokenizer)

mean_bleu = sum(bleu) / len(bleu)
print(mean_bleu)

#with('final_bleu_score.txt', 'w') as f:
 #   f.write(f'Bleu score is {mean_bleu}')

