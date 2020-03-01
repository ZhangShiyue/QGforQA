# QGforQA (EMNLP 2019)

This repository contains source code for the systems described in:

[Addressing Semantic Drift in Question Generation for Semi-Supervised Question Answering](https://arxiv.org/abs/1909.06356)

## Modules
[ELMo-QG](QG/ELMo_QG), [BiDAF-QA](QA/BiDAF_QA), [ELMo-QPC](QPC/ELMo_QPC)

[BERT-QG](QG/BERT_QG), [BERT-QA](QA/BERT_QA), [BERT-QPC](QPC/BERT_QPC) (Code is coming soon...)

## Dependencies
### Python

The code requires Python 3. Some basic python dependencies are specified in "requirement.txt".
```
pip install -r requirements.txt
```
By the way, a Python 3 virtual environment could be set up and run with:
```
virtualenv name_of_environment -p python3
source name_of_environment/bin/activate
```

### Standford CoreNLP

Setup [Standford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) environment by running the following commands:
```
mkdir LIB/corenlp
cd LIB/corenlp; wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip; unzip stanford-corenlp-full-2018-10-05.zip
export CORENLP_HOME=$PWD/stanford-corenlp-full-2018-10-05
```

### GloVe

Download [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings:
```
mkdir LIB/glove
cd LIB/glove; wget http://nlp.stanford.edu/data/glove.840B.300d.zip; unzip glove.840B.300d.zip
```

### ELMo

Download [ELMo](https://allennlp.org/elmo):
```
mkdir LIB/elmo
cd LIB/elmo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
```
Setup ELMo environment:
```
git clone https://github.com/allenai/bilm-tf.git
cd bilm-tf; python setup.py install
```

### BERT

Download [BERT](https://github.com/google-research/bert):
```
cd LIB/bert
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

### SQuADv1.1

Download [SQuADv1.1](https://arxiv.org/abs/1606.05250) dataset:
```
mkdir LIB/squad
cd LIB/squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

Download the article title list of [SQuAD QG](https://github.com/xinyadu/nqg) test set:
```
wget https://raw.githubusercontent.com/xinyadu/nqg/master/data/doclist-test.txt
```

### QQP

Download [QQP](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset:
```
python LIB/download.py
```

