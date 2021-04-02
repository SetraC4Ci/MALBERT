"""Dataset Class for Bert"""
import random
import tqdm
import torch
import linecache
from torch.utils.data import Dataset
from .tokenizer import BertTokenizer


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer: BertTokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.corpus_lines = sum(1 for line in open(self.corpus_path))

        # with open(corpus_path, "r", encoding=encoding) as f:
        #     if self.corpus_lines is None and not on_memory:
        #         for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
        #             self.corpus_lines += 1

        #     if on_memory:
        #         self.lines = [line[:-1].split("\t")
        #                       for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
        #         self.corpus_lines = len(self.lines)

        # if not on_memory:
        #     self.file = open(corpus_path, "r", encoding=encoding)
        #     self.random_file = open(corpus_path, "r", encoding=encoding)

        #     for _ in range(random.randint(0, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.tokenizer.sos_index] + t1_random + [self.tokenizer.eos_index]
        t2 = t2_random + [self.tokenizer.eos_index]

        t1_label = [self.tokenizer.pad_index] + t1_label + [self.tokenizer.pad_index]
        t2_label = t2_label + [self.tokenizer.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.tokenizer.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()} #pylint: disable=not-callable

    def random_word(self, sentence):
        # tokens = sentence.split()
        output_label = []
        tokens = self.tokenizer.tokenize(sentence)
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    tokens[i] = self.tokenizer.mask_index
                elif prob < 0.9:
                    tokens[i] = self.tokenizer.getRandomTokenID()
                else:
                    tokens[i] = token
                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(0)
        return tokens, output_label

        # for i, token in enumerate(tokens):
        #     prob = random.random()
        #     if prob < 0.15:
        #         prob /= 0.15

        #         # 80% randomly change token to mask token
        #         if prob < 0.8:
        #             tokens[i] = self.vocab.mask_index

        #         # 10% randomly change token to random token
        #         elif prob < 0.9:
        #             tokens[i] = random.randrange(len(self.vocab))

        #         # 10% randomly change token to current token
        #         else:
        #             tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        #         output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

        #     else:
        #         tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
        #         output_label.append(0)

        # return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        # t1 = self.tokenizer.tokenize(t1)
        # t2 = self.tokenizer.tokenize(t2)
        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            # rand_line = self.tokenizer.tokenize(self.get_random_line())
            return t1, self.get_random_line(), 0

    # def get_corpus_line(self, item):
    #     if self.on_memory:
    #         return self.lines[item][0], self.lines[item][1]
    #     else:
    #         line = self.file.__next__()
    #         if line is None:
    #             self.file.close()
    #             self.file = open(self.corpus_path, "r", encoding=self.encoding)
    #             line = self.file.__next__()

    #         t1, t2 = line[:-1].split("\t")
    #         return t1, t2
    def get_corpus_line(self, item):
        t1 = linecache.getline(self.corpus_path, item)
        t2 = linecache.getline(self.corpus_path, item+1)
        return t1, t2

    # def get_random_line(self):
    #     if self.on_memory:
    #         return self.lines[random.randrange(len(self.lines))][1]

    #     line = self.file.__next__()
    #     if line is None:
    #         self.file.close()
    #         self.file = open(self.corpus_path, "r", encoding=self.encoding)
    #         for _ in range(random.randint(0, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
    #             self.random_file.__next__()
    #         line = self.random_file.__next__()
    #     return line[:-1].split("\t")[1]

    def get_random_line(self):
        return linecache.getline(self.corpus_path, random.randint(1, self.corpus_lines))