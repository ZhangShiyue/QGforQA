## ELMo-based QG model

### Quick Start (test pre-trained models)

#### Download data and pre-trained models
Download our [experimental data](https://drive.google.com/open?id=1Fje3uD6mOvV6mJRGStQ7-NzRixG-Q6-c) and [vocabulary](https://drive.google.com/open?id=1pmeZ6Am8hYf6ggFMlTwQMbg2dk8vOrf8), and put them under "data" directory.

Download our [pre-trained models](https://drive.google.com/open?id=1YW-mbiklsGsC-tJOoDUJxj1JbnRFQQnZ) which contains four models: our ELMo-QG baseline, QPP-reinforced model, 
QAP-reinforced model and QPP&QAP-reinforced model.

#### Test pre-trained models
```
python config.py --mode test \
    --output_dir train/$MODEL \
    --best_ckpt train/$MODEL/best_ckpt.json
    --beam_size 10 
```
"$MODEL" can be "model_elmo_qg" (for baseline) or "model_elmo_qg_qpp" (for QPP), etc.


### From scratch 
#### Preprocess
Run the following command to tokenize and obtain POS/NER tags for SQuAD QG dataset.
```
python preprocess.py
```
Here is our [preprocessed data](https://drive.google.com/file/d/1qCo2pK5iBGjQhYJ_EhrQDw8exD2EDUCa/view?usp=sharing).

#### Get vocabulary
Run the following command to obtain the word/pos/ner/label dictionaries/embeddings used in our QG systems.
Two word embeddings from GloVe and ELMo will be saved. Pos/ner/label embeddings will be randomly 
initialized. All vocabulary related files will save in "data/vocab" directory automatically.
```
python config.py --mode get_vocab
```
Here is our [vocabulary](https://drive.google.com/open?id=1pmeZ6Am8hYf6ggFMlTwQMbg2dk8vOrf8). Note that, you can't test our pre-trained models on newly obtained vocabulary, 
because the unstable sort will make vocabulary indexes different. 


#### Prepare data
Run the following command to prepare experimental data which is saved as TFrecords in "data/experimental" directory.
```
python config.py --mode prepare
```

#### Train
1. Train our ELMo-QG baseline model:
```
python config.py --mode train
```

2. Train our BLEU/METEOR/ROUGE reinforced model:

Download the [pre-trained ELMo-QG baseline model](https://drive.google.com/open?id=113SChSTRu1OnwaMjWgqXjPiESWYV1ooO), and put it under "train/model_elmo_qg_rl" directory.
```
python config.py --mode train_rl \
    --output_dir train/model_elmo_qg_rl  \
    --best_ckpt train/model_elmo_qg_rl/best_ckpt.json \
    --rl_metric $METRIC
    --beam_size 1
```
$METRIC can be "bleu", "rouge" or "meteor".

3. Train our QPP reinforced model:

Download the [pre-trained ELMo-QG baseline model](https://drive.google.com/open?id=113SChSTRu1OnwaMjWgqXjPiESWYV1ooO), and put it under "train/model_elmo_qg_qpp" directory.

Download the [pre-trained ELMO-QPC model](https://drive.google.com/open?id=1_ukdkwcrlKOR9NPW1RbFvEUsjqRxZKxd), and put it under "../../QPC/ELMo_QPC/" directory.
```
python config.py --mode train_qpp \
    --output_dir train/model_elmo_qg_qpp  \
    --best_ckpt train/model_elmo_qg_qpp/best_ckpt.json \
    --output_dir_qpc ../../QPC/ELMo_QPC/train/model_elmo_qpc \
    --best_ckpt_qpc ../../QPC/ELMo_QPC/train/model_elmo_qpc/best_ckpt.json
    --beam_size 1
```

4. Train our QAP reinforced model:

Download the [pre-trained ELMo-QG baseline model](https://drive.google.com/open?id=113SChSTRu1OnwaMjWgqXjPiESWYV1ooO), and put it under "train/model_elmo_qg_qap" directory.

Download the [pre-trained BiDAF-QA model](https://drive.google.com/open?id=16Wt2hboUtKb3QLTyg5g-ZOLr9QWRXSA1), and put it under "../../QA/BiDAF_QA/" directory.
```
python config.py --mode train_qap \
    --output_dir train/model_elmo_qg_qap  \
    --best_ckpt train/model_elmo_qg_qap/best_ckpt.json \
    --output_dir_qa ../../QA/BiDAF_QA/train/model_bidaf_qa_for_qg \
    --best_ckpt_qa ../../QA/BiDAF_QA/train/model_bidaf_qa_for_qg/best_ckpt.json
    --beam_size 1
```

5. Train our QPP & QAP reinforced model:

Download the [pre-trained ELMo-QG baseline model](https://drive.google.com/open?id=113SChSTRu1OnwaMjWgqXjPiESWYV1ooO), and put it under "train/model_elmo_qg_qpp_qap" directory.

Download the [pre-trained ELMO-QPC model](https://drive.google.com/open?id=1_ukdkwcrlKOR9NPW1RbFvEUsjqRxZKxd), and put it under "../../QPC/ELMo_QPC/" directory.

Download the [pre-trained BiDAF-QA model](https://drive.google.com/open?id=16Wt2hboUtKb3QLTyg5g-ZOLr9QWRXSA1), and put it under "../../QA/BiDAF_QA/" directory.
```
python config.py --mode train_qpp_qap \
    --output_dir train/model_elmo_qg_qpp_qap  \
    --best_ckpt train/model_elmo_qg_qpp_qap/best_ckpt.json \
    --output_dir_qpc ../../QPC/ELMo_QPC/train/model_elmo_qpc \
    --best_ckpt_qpc ../../QPC/ELMo_QPC/train/model_elmo_qpc/best_ckpt.json \
    --output_dir_qa ../../QA/BiDAF_QA/train/model_bidaf_qa_for_qg \
    --best_ckpt_qa ../../QA/BiDAF_QA/train/model_bidaf_qa_for_qg/best_ckpt.json
```

#### Test
```
python config.py --mode test \
    --output_dir train/$MODEL \
    --best_ckpt train/$MODEL/best_ckpt.json
    --beam_size 10 
```

### TODO
1. API for generating questions




