""" Tokenizer class """
import os
import random
from pathlib import Path
import tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Digits


class BertTokenizer():
    """Bert Tokenizer using WordPiece Tokenizer Model"""
    def __init__(self, path):
        self.path = path
        text_paths = [str(x) for x in Path("./dataset/corpus/").glob("**/*.txt")]
        savedpath = "./dataset/tok_model/MaLaMo-vocab.txt"
        if os.path.exists(savedpath):
            self.tokenizer = tokenizers.BertWordPieceTokenizer(
                "./dataset/tok_model/MaLaMo-vocab.txt",
            )
        else:
            self.tokenizer = tokenizers.BertWordPieceTokenizer()
            self.tokenizer.train(files=text_paths, special_tokens=[
                                 "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=14200)
            self.tokenizer.save_model("./dataset/tok_model", "MaLaMo")
        self.tokenizer.enable_truncation(max_length=512)
        self.pretokenizer = tokenizers.pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        self.vocab = self.tokenizer.get_vocab()
        self.mask_index = self.vocab.get("[MASK]")
        self.pad_index = self.vocab.get("[PAD]")
        self.eos_index = self.vocab.get("[SEP]")
        self.sos_index = self.vocab.get("[CLS]")
        self.unk_index = self.vocab.get("[UNK]")

    
    def tokenize(self, sentence: str):
        return self.tokenizer.encode(sentence).ids
    
    def getRandomTokenID(self):
        return random.randint(6, len(self.vocab) - 1)
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
        