"""main entry for training"""

import argparse

from torch.utils.data import DataLoader

from model.bert import BERT
from trainer import BERTTrainer
from dataset import BERTDataset, BertTokenizer


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", type=str, default="./dataset/corpus/train.txt", help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="./dataset/corpus/test.txt", help="test set for evaluate train set")
    #parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", type=str, default="./output/bert.model",  help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=512, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=8, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=1, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=5110, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=False, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    print("Loading Vocab")
    tokenizer = BertTokenizer("./dataset/corpus")
    vocab_size = tokenizer.get_vocab_size()
    print("Vocab Size: ", vocab_size)

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, tokenizer, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, tokenizer, seq_len=args.seq_len, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(vocab_size, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, vocab_size, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    train()
