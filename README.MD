# MALBERT

Malagasy Langage BERT - Strongly inspired by [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch) but using pytorch integrated transformer module


## Quickstart

**NOTICE : Your corpus should be one sentence per line

### 0. Prepare your corpus
Put train.txt, test.txt, valid.txt in folder dataset/corpus

### 1. Pretrain model
```
$python3 main.py 
```

## Dependencies
* python 3.8
* torch >= 1.4
* tokenizers


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)