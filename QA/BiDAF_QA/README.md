## BiDAF QA model

### Quick Start (test pre-trained models)

#### Test the pre-trained QA model for QG (to obtain QAP)
Link to "data" directory of ELMo-QG:
```
ln -s ../../QG/ELMo_QG/data data
```
Download our [pre-trained models](https://drive.google.com/open?id=16Wt2hboUtKb3QLTyg5g-ZOLr9QWRXSA1). 
```
python config.py --mode test_qa_for_qg \
    --output_dir train/model_bidaf_qa_for_qg  \
    --best_ckpt train/model_bidaf_qa_for_qg/best_ckpt.json \
```

### From scratch 






