## BERT-based QG model

### Quick Start (test pre-trained models)

#### Download data and pre-trained models
Download our [experimental data](https://drive.google.com/open?id=1PWg-amXBRavHzhZLbFZ7Eb9za__Tf4Yg) and [vocabulary](https://drive.google.com/open?id=1PbnwFvFwnnXE_tVnP1IxS_dvaEsBOcF5), and put them under "data" directory.

Download our [pre-trained models](https://drive.google.com/open?id=1gCOA6eNF25BKtRkurgWWY1Gjj6HwTPz1) which contains four models: our BERT-QG baseline, QPP-reinforced model, 
QAP-reinforced model and QPP&QAP-reinforced model.

#### Test pre-trained models
```
python config.py --mode test \
    --output_dir train/$MODEL \
    --best_ckpt train/$MODEL/best_ckpt.json
    --beam_size 10 
```
"$MODEL" can be "model_bert_qg" (for baseline) or "model_bert_qg_qap" (for QAP), etc.


### From scratch 
#### Preprocess
Download the [ELMo-QG preprocessed data](https://drive.google.com/file/d/1qCo2pK5iBGjQhYJ_EhrQDw8exD2EDUCa/view?usp=sharing).
See the [README](../ELMo_QG/README.md) of ELMo-QG for how to obtain this data.

Run the following command to bert-tokenize ELMo-QG processed data.
```
python preprocess.py
```
Here is our [preprocessed data](https://drive.google.com/open?id=1jWo4ZI7aY5FP7Qboqa5AOVsrxg2JS4LX).


#### Get vocabulary
Download [vocabulary](https://drive.google.com/open?id=1PbnwFvFwnnXE_tVnP1IxS_dvaEsBOcF5) for BERT-QG.


#### Prepare data
Run the following command to prepare experimental data which is saved as TFrecords in "data/experimental" directory.
```
python config.py --mode prepare
```

#### Train
1. Train our BERT-QG baseline model:
```
python config.py --mode train
```

### TODO
1. BERT-QA
2. BERT-QPC
2. reinforced models