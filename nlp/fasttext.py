import csv
import fasttext
import pandas as pd
import numpy as np

from config import SEED
from nlp.classification_model import Model, PREDICT_PROBA_N

FASTTEXT_LABEL_PREFIX = '__label__'

class FastText(Model):

    def __init__(self):
        self.model = None

    def train(self, X, y):
        # fasttext library requires a file as input; labels are identified by the '__label__' prefix
        tmp_file = 'fasttext.train'
        pd.DataFrame([y.apply(lambda lbl: FASTTEXT_LABEL_PREFIX+lbl), X]).T\
            .to_csv(tmp_file, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar="")
        # thread 1 required for reproducible results -> https://fasttext.cc/docs/en/faqs.html
        self.model = fasttext.train_supervised(tmp_file, seed=SEED, thread=1)

    def is_trained(self):
        return self.model is not None

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        lbls, ps = self.model.predict(X, k=n)
        lbls = [lbl[len(FASTTEXT_LABEL_PREFIX):] for lbl in lbls]  # remove '__label__' prefix
        return np.stack([lbls, ps], axis=1)

    def get_dumped_model_path(self):
        tmp_file = 'fasttext.bin'
        self.model.save_model(tmp_file)
        return tmp_file
