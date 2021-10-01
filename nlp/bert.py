from pathlib import Path

import pandas as pd
import numpy as np
import torch
import shutil

from nlp.classification_model import Model, PREDICT_PROBA_N
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy, set_seed
from datasets import Dataset

BASE_MODEL = 'distilbert-base-uncased'
USE_EARLY_STOPPING = True
USE_DUMMY_BERT = False  # stops after a few epochs, used for tests


class Bert(Model):

    def __init__(self, seed=None):
        super(Bert, self).__init__(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = None
        self.idx_to_label = None  # list of target labels, according to trained class indices
        self.train_only_clf_layer = True  # don't retrain the whole model but only the clf layer at the end
        self.batch_size = 8
        self.max_train_steps = 1  # only used if USE_DUMMY_BERT is True

    def train(self, X, y):

        if USE_DUMMY_BERT:
            # reduce size of training set to reduce number of batches
            X = X[:self.batch_size*self.max_train_steps]
            y = y[:self.batch_size*self.max_train_steps]

        self.idx_to_label = sorted(list(set(y)))
        dataset = Dataset.from_pandas(pd.DataFrame({
            'text': X,
            'label': [self.idx_to_label.index(lbl_str) for lbl_str in y]}))

        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        def model_init():
            set_seed(self.seed)  # from_pretrained seems to be flawed regarding seed usage
            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=len(self.idx_to_label))
            if self.train_only_clf_layer:
                self._set_requires_grad(model, False)
            return model

        training_args = TrainingArguments("test_trainer", seed=self.seed, per_device_train_batch_size=self.batch_size,
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
        set_seed(self.seed)
        inputs = self.tokenizer(X, return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits.tolist()
        probs = self.softmax(logits)
        ret_idx = (-1*probs).argsort()[:,:n]
        ps_ret = probs[np.repeat(np.arange(len(probs)), n), ret_idx.flatten()].reshape(len(probs), n)
        return np.stack([np.array(self.idx_to_label)[ret_idx], ps_ret], axis=2)

    def get_dumped_model_path(self):
        tmp_folder = Path('bert_model')
        zip_file = Path('bert_model.zip')
        self.model.save_pretrained(str(tmp_folder))
        self.tokenizer.save_pretrained(str(tmp_folder))
        shutil.make_archive(str(tmp_folder), 'zip', str(tmp_folder))
        return zip_file.resolve()

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)