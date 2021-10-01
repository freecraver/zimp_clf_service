import csv
import fasttext
import pandas as pd
import numpy as np

from nlp.classification_model import Model, PREDICT_PROBA_N
from pathlib import Path

FASTTEXT_LABEL_PREFIX = '__label__'


class FastText(Model):

    def __init__(self, seed=None):
        super(FastText, self).__init__(seed)
        self.model = None

    def train(self, X, y):
        # fasttext library requires a file as input; labels are identified by the '__label__' prefix
        tmp_file = 'fasttext.train'
        pd.DataFrame([y.apply(lambda lbl: FASTTEXT_LABEL_PREFIX+lbl), X]).T\
            .to_csv(tmp_file, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar="")
        # thread 1 required for reproducible results -> https://fasttext.cc/docs/en/faqs.html
        self.model = fasttext.train_supervised(tmp_file, seed=self.seed, thread=1)

    def is_trained(self):
        return self.model is not None

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        lbls_list, ps = self.model.predict(X, k=n)
        for idx, lbls in enumerate(lbls_list):
            lbls_list[idx] = [lbl[len(FASTTEXT_LABEL_PREFIX):] for lbl in lbls]  # remove '__label__' prefix
        return np.stack([lbls_list, ps], axis=2)

    def get_dumped_model_path(self):
        tmp_file = Path('fasttext.bin')
        self.model.save_model(str(tmp_file))
        return tmp_file.resolve()
