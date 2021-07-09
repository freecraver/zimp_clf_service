from pathlib import Path

import pandas as pd
import numpy as np
import torch
import shutil

from config import SEED
from nlp.classification_model import Model, PREDICT_PROBA_N
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy, set_seed
from datasets import Dataset

BASE_MODEL = 'distilbert-base-uncased'
USE_EARLY_STOPPING = False
USE_DUMMY_BERT = True  # stops after a few epochs, used for tests

class Bert(Model):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = None
        self.idx_to_label = None  # list of target labels, according to trained class indices
        self.train_only_clf_layer = True  # don't retrain the whole model but only the clf layer at the end
        self.batch_size = 8
        self.max_train_steps = 5  # only used if USE_DUMMY_BERT is True

    def train(self, X, y):

        if USE_DUMMY_BERT:
            # reduce size of training set to reduce number of batches
            X = X[:self.batch_size*self.max_train_steps]
            y = y[:self.batch_size*self.max_train_steps]

        self.idx_to_label = list(set(y))
        dataset = Dataset.from_pandas(pd.DataFrame({
            'text': X,
            'label': [self.idx_to_label.index(lbl_str) for lbl_str in y]}))

        set_seed(SEED)

        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        def model_init():
            set_seed(SEED)  # from_pretrained seems to be flawed regarding seed usage
            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=len(self.idx_to_label))
            if self.train_only_clf_layer:
                self._set_requires_grad(model, False)
            return model

        training_args = TrainingArguments("test_trainer", seed=SEED, per_device_train_batch_size=self.batch_size,
                                          logging_steps=1)
        if USE_EARLY_STOPPING:
            training_args.load_best_model_at_end = True
            training_args.evaluation_strategy = IntervalStrategy.EPOCH
            training_args.metric_for_best_model = 'accuracy'
        if USE_DUMMY_BERT:
            training_args.max_steps = self.max_train_steps

        trainer = Trainer(model_init=model_init, args=training_args, train_dataset=tokenized_dataset)

        if USE_EARLY_STOPPING:
            trainer.add_callback(EarlyStoppingCallback())

        trainer.train()
        self.model = trainer.model

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    @staticmethod
    def _set_requires_grad(model, requires_grad):
        for param in model.base_model.parameters():
            param.requires_grad = requires_grad

    def is_trained(self):
        return self.model is not None

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        set_seed(SEED)
        inputs = self.tokenizer(X, return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits.tolist()
        probs = self.softmax(logits)[0]
        ret_idx = (-1*probs).argsort()[:n]
        return np.stack([np.array(self.idx_to_label)[ret_idx], probs[ret_idx]], axis=1)

    def get_dumped_model_path(self):
        tmp_folder = Path('bert_model')
        zip_file = Path('bert_model.zip')
        self.model.save_pretrained(str(tmp_folder))
        shutil.make_archive(str(tmp_folder), 'zip', str(tmp_folder))
        return zip_file.resolve()

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1)