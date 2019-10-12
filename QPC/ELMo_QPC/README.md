## ELMo-based QPC model

### Quick Start (test the pre-trained model)

#### Download data and the pre-trained model
Download our [experimental data](https://drive.google.com/file/d/1mjYgr5kXBDrYaHL0IeTBdK1xU42PqNB1/view?usp=sharing) and [vocabulary](https://drive.google.com/open?id=1pmeZ6Am8hYf6ggFMlTwQMbg2dk8vOrf8), 
and put them under "data" directory.

Download our [pre-trained model](https://drive.google.com/open?id=1_ukdkwcrlKOR9NPW1RbFvEUsjqRxZKxd).

#### Test the pre-trained model
```
python config.py --mode test \
    --output_dir train/model_elmo_qpc \
    --best_ckpt train/model_elmo_qpc/best_ckpt.json
```


### From scratch 
#### Preprocess
In this step, the QQP dataset will be tokenized.
```
python preprocess.py
```

#### Get vocabulary
Here, we use the same vocabulary as being used for ELMo-QG. Download the vocabulary [here](https://drive.google.com/open?id=1pmeZ6Am8hYf6ggFMlTwQMbg2dk8vOrf8) 
or link to ELMo-QG's vocabulary.

#### Prepare data
Run the following command to get experimental data 
which is saved as TFrecords in "data/experimental" directory.
```
python config.py --mode prepare
```

#### Train
Train our ELMo-QPC model:
```
python config.py --mode train
```

#### Test
```
python config.py --mode test \
    --output_dir train/model_elmo_qpc \
    --best_ckpt train/model_elmo_qpc/best_ckpt.json
```






