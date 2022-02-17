import csv
import fasttext
import pandas as pd
import numpy as np
import re

from nlp.classification_model import Model, PREDICT_PROBA_N
from pathlib import Path
from sklearn.utils import shuffle

FASTTEXT_LABEL_PREFIX = '__label__'
ENABLE_PRE_PROCESSING = False


class FastText(Model):

    def __init__(self, seed=None, **kwargs):
        super(FastText, self).__init__(seed)
        self.model = None
        self.kwargs = kwargs

    def train(self, X, y):
        # first shuffle data as fasttext uses SGD (https://github.com/facebookresearch/fastText/issues/74)
        X, y = shuffle(X, y, random_state=self.seed)

        # fasttext library requires a file as input; labels are identified by the '__label__' prefix
        tmp_file = 'fasttext.train'
        X = self.pre_process_text(X)

        pd.DataFrame([y.apply(lambda lbl: FASTTEXT_LABEL_PREFIX+str(lbl)), X]).T\
            .to_csv(tmp_file, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar="")
        # thread 1 required for reproducible results -> https://fasttext.cc/docs/en/faqs.html
        # for some models a higher learning rate will be required
        self.model = fasttext.train_supervised(tmp_file, seed=self.seed, thread=1, **self.kwargs)

    def is_trained(self):
        return self.model is not None

    @staticmethod
    def pre_process_text(X):
        if ENABLE_PRE_PROCESSING:
            return X.apply(lambda txt: re.sub(r'\W', ' ', txt))  # remove all non-text chars
        else:
            # tabs must be removed, otherwise training fails
            return X.apply(lambda txt: txt.replace('\t', ' ').replace('\n', ' '))

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        X = self.pre_process_text(pd.Series(X)).to_list() # remove all non-text chars which fasttext won't use
        lbls_list, ps = self.model.predict(X, k=n)
        for idx, lbls in enumerate(lbls_list):
            lbls_list[idx] = [lbl[len(FASTTEXT_LABEL_PREFIX):] for lbl in lbls]  # remove '__label__' prefix
        return np.stack([lbls_list, ps], axis=2)

    def get_dumped_model_path(self):
        tmp_file = Path('fasttext.bin')
        self.model.save_model(str(tmp_file))
        return tmp_file.resolve()
