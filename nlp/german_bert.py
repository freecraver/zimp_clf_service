import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed, AutoModelForSequenceClassification

from nlp.bert import Bert, USE_DUMMY_BERT

BASE_MODEL = 'bert-base-german-cased'


class GermanBert(Bert):

    def get_base_model(self):
        return BASE_MODEL

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.get_base_model())

    def init_model(self, training_args):
        def f_model_init():
            set_seed(self.seed)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.get_base_model(),
                num_labels=len(self.idx_to_label),
                **self.kwargs
            )
            if self.train_only_clf_layer:
                self._set_requires_grad(model, False)
            return model
        return f_model_init

    def get_max_steps(self, cnt_train_records):
        return min(12000, int(cnt_train_records * 50 / self.batch_size))

    def init_dataset(self, X, y):
        dataset = Dataset.from_pandas(pd.DataFrame({
            'text': X,
            'label': [self.idx_to_label.index(lbl_str) for lbl_str in y]}))

        return dataset.map(self._tokenize_function, batched=True)

    def init_args(self, max_steps, model_path):
        return TrainingArguments(model_path, seed=self.seed, per_device_train_batch_size=self.batch_size,
                                 max_steps=max_steps, logging_steps=100, no_cuda=USE_DUMMY_BERT)

    def init_trainer(self, model_init, args, dataset):
        return Trainer(model_init=model_init, args=args, train_dataset=dataset)

    def _set_requires_grad(self, model, requires_grad):
        for param in model.base_model.parameters():
            param.requires_grad = requires_grad

    def predict_logits(self, X):
        inputs = self.tokenizer(X, return_tensors="pt", padding="max_length", truncation=True)
        if not USE_DUMMY_BERT:
            inputs = inputs.to('cuda')
        with torch.no_grad():
            logits = self.model(**inputs).logits
            if not USE_DUMMY_BERT:
                logits = logits.to('cpu')
            return logits.numpy()

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

