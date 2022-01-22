import numpy as np
import tensorflow as tf
import os
import shutil
import transformers

from pathlib import Path

from sklearn.utils import shuffle

from nlp.classification_model import Model, PREDICT_PROBA_N
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, TFTrainingArguments, TFTrainer
from transformers.trainer_utils import set_seed

BASE_MODEL = 'distilbert-base-uncased'
USE_DUMMY_BERT = False  # stops after a few training steps, used for tests


class Bert(Model):

    def __init__(self, seed=None):
        super(Bert, self).__init__(seed)
        transformers.logging.set_verbosity_info()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
        self.model = None
        self.idx_to_label = None  # list of target labels, according to trained class indices
        self.train_only_clf_layer = True  # don't retrain the whole model but only the clf layer at the end
        self.batch_size = 4
        self.max_train_steps = 1  # only used if USE_DUMMY_BERT is True

    def train(self, X, y):
        X, y = shuffle(X, y, random_state=self.seed)

        if USE_DUMMY_BERT:
            # reduce size of training set to reduce number of batches
            X = X[:self.batch_size * self.max_train_steps]
            y = y[:self.batch_size * self.max_train_steps]

        self.idx_to_label = sorted(list(set(y)))
        train_labels = [self.idx_to_label.index(lbl_str) for lbl_str in y]
        train_encodings = self.tokenizer(X.tolist(), return_tensors='tf', padding=True, truncation=True)
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        ))

        max_steps = min(10000, 3*len(X))

        model_path = 'test_trainer'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)  # huggingface overwrite dir does not work as specified, perform hard delete
        training_args = TFTrainingArguments(model_path, seed=self.seed, per_device_train_batch_size=self.batch_size,
                                            logging_steps=100, max_steps=max_steps)
        if USE_DUMMY_BERT:
            training_args.max_steps = self.max_train_steps

        with training_args.strategy.scope():
            set_seed(self.seed)
            model = TFDistilBertForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=len(self.idx_to_label))

        if self.train_only_clf_layer:
            self._set_requires_grad(model, False)

        trainer = TFTrainer(model=model, args=training_args, train_dataset=dataset)

        trainer.train()
        self.model = trainer.model

    @staticmethod
    def _set_requires_grad(model, requires_grad):
        for layer in model.layers:
            if layer.name not in ['distilbert']:
                continue
            layer.trainable = requires_grad

    def is_trained(self):
        return self.model is not None

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        set_seed(self.seed)
        inputs = self.tokenizer(X, return_tensors="tf", padding=True, truncation=True)
        logits = self.model(**inputs).logits.numpy()
        probs = self.softmax(logits)
        ret_idx = (-1 * probs).argsort()[:, :n]
        ps_ret = probs[np.repeat(np.arange(len(probs)), n), ret_idx.flatten()].reshape(len(probs), n)
        return np.stack([np.array(self.idx_to_label, dtype=np.str)[ret_idx], ps_ret], axis=2)

    def get_dumped_model_path(self):
        tmp_folder = Path('bert_model')
        zip_file = Path('bert_model.zip')
        self.model.save_pretrained(str(tmp_folder))
        self.tokenizer.save_pretrained(str(tmp_folder))
        shutil.make_archive(str(tmp_folder), 'zip', str(tmp_folder))
        return zip_file.resolve()

    def get_prediction_batch_size(self) -> int:
        return 6

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)