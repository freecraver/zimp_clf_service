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

    def __init__(self, seed=None, **kwargs):
        super(Bert, self).__init__(seed)
        transformers.logging.set_verbosity_info()
        self.model = None
        self.idx_to_label = None  # list of target labels, according to trained class indices
        self.train_only_clf_layer = True  # don't retrain the whole model but only the clf layer at the end
        self.batch_size = 4
        self.tokenizer = self.init_tokenizer()
        self.kwargs = kwargs

    def get_base_model(self):
        return BASE_MODEL

    def init_tokenizer(self):
        return DistilBertTokenizerFast.from_pretrained(self.get_base_model())

    def init_model(self, training_args):
        with training_args.strategy.scope():
            set_seed(self.seed)
            model = TFDistilBertForSequenceClassification.from_pretrained(
                self.get_base_model(),
                num_labels=len(self.idx_to_label),
                **self.kwargs
            )

        if self.train_only_clf_layer:
            self._set_requires_grad(model, False)
        return model

    def get_max_steps(self, cnt_train_records):
        return min(6000, int(cnt_train_records * 50 / self.batch_size))

    def train(self, X, y):
        X, y = shuffle(X, y, random_state=self.seed)

        if USE_DUMMY_BERT:
            # reduce size of training set to reduce number of batches
            X = X[:self.batch_size]
            y = y[:self.batch_size]

        self.idx_to_label = sorted(list(set(y)))
        dataset = self.init_dataset(X, y)

        max_steps = 1 if USE_DUMMY_BERT else self.get_max_steps(len(X))  # only 1 step for tests

        model_path = 'test_trainer'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)  # huggingface overwrite dir does not work as specified, perform hard delete
        training_args = self.init_args(max_steps, model_path)

        model = self.init_model(training_args)

        trainer = self.init_trainer(model, training_args, dataset)
        trainer.train()
        self.model = trainer.model

    def _set_requires_grad(self, model, requires_grad):
        for layer in model.layers:
            if layer.name not in ['distilbert']:
                continue
            layer.trainable = requires_grad

    def init_dataset(self, X, y):
        train_labels = [self.idx_to_label.index(lbl_str) for lbl_str in y]
        train_encodings = self.tokenizer(X.tolist(), return_tensors='tf', padding=True, truncation=True)
        return tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        ))

    def init_args(self, max_steps, model_path):
        return TFTrainingArguments(model_path, seed=self.seed, per_device_train_batch_size=self.batch_size,
                                   logging_steps=100, max_steps=max_steps, no_cuda=USE_DUMMY_BERT)

    def init_trainer(self, model, args, dataset):
        return TFTrainer(model=model, args=args, train_dataset=dataset)

    def is_trained(self):
        return self.model is not None

    def predict_logits(self, X):
        inputs = self.tokenizer(X, return_tensors="tf", padding=True, truncation=True)
        return self.model(**inputs).logits.numpy()

    def predict_proba(self, X, n=PREDICT_PROBA_N):
        set_seed(self.seed)
        logits = self.predict_logits(X)
        probs = self.softmax(logits)
        return self.transform_prediction_output(probs, n, np.array(self.idx_to_label, dtype=np.str))

    def get_dumped_model_path(self):
        tmp_folder = Path('bert_model')
        zip_file = Path('bert_model.zip')
        self.model.save_pretrained(str(tmp_folder))
        self.tokenizer.save_pretrained(str(tmp_folder))
        shutil.make_archive(str(tmp_folder), 'zip', str(tmp_folder))
        return zip_file.resolve()

    def get_prediction_batch_size(self) -> int:
        return 128