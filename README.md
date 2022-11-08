# LMSD
## Requirements
[Learning Metric Space with Distillation for Large-Scale Multi-Label Text Classification]
* python==3.8.8
* tensorflow==2.7.0

## Datasets
* [Reuter](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection)
* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [CiteULike-t](https://citeulike.org/faq/data.adp/)
* Download the GloVe embedding (840B,300d)  (https://nlp.stanford.edu/projects/glove/)

## Train and Test
Run main.py for train and test datasets with tokenized texts as follows:
```bash
python main.py --data 'Wiki10-31K' --la 1.0 --ba 1.0 --emb_num 256 --n_negative 1000
```

