# LMSD
## Requirements

* python==3.8.8
* tensorflow==2.7.0

## Datasets
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* Download the GloVe embedding (840B,300d)  (https://nlp.stanford.edu/projects/glove/)

## Run
python main.py --data 'Wiki10-31K' --la 1.0 --ba 1.0 --emb_num 256 --n_negative 100



