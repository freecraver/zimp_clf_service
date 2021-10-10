"""
This is very lightweight classification service used for training and evaluating models under different scenarios
in a containerized env
"""

from flask import Flask, jsonify, request, send_file, redirect
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.middleware.proxy_fix import ProxyFix

import pandas as pd
import json

from nlp.classification_provider import ClassificationProvider, ModelType

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


@app.route('/train', methods=['POST'])
def train():
    """
    Trains a model using a csv file with ',' delimiters and the column headers 'text', 'target'
    May take several minutes, depending on used model
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        description: The file to upload.
      - in: formData
        name: modelType
        type: string
        enum: [SVM, FASTTEXT, BERT]
        description: Type of model to be trained
      - in: formData
        name: seed
        type: integer
        description: random seed required for reproducibility
      - in: formData
        name: asynchronous
        type: boolean
        description: do not wait for training to complete
    responses:
      400:
        description: Required file missing or having invalid format
      200:
        description: Model successfully trained
    """
    if 'file' not in request.files:
        return 'File upload required', 400

    model_type = request.form.get('modelType')

    if not ModelType.has_value(model_type):
        return 'Please supply valid model type', 400

    df = pd.read_csv(request.files.get('file'))
    if df.shape[1] != 2 or len(set(df.columns) - {'target', 'text'}) > 0:
        return 'Supplied invalid header columns. Must be text, target', 400

    seed = int(request.form.get('seed')) if request.form.get('seed') else None
    model = ClassificationProvider().init_model(ModelType(model_type), seed)

    if json.loads(request.form.get('asynchronous', 'false')):
        model.train_async(df['text'], df['target'])
        return 'Training started', 200

    model.train(df['text'], df['target'])
    return 'Model trained', 200


@app.route("/predict", methods=['POST'])
def predict():
    """
    Predict the class label of one input text. Requires a previous train-call
    ---
    parameters:
      - in: body
        name: body
        schema:
          required:
            - text
          properties:
            text:
              type: string
    responses:
      400:
        description: Invalid Text or model not previously trained
      200:
        description: Predicted class label
        schema:
          required:
            - label
          properties:
            label:
              type: string
    """
    data = request.get_json()
    if data is None or 'text' not in data:
        return 'Missing text in body', 400
    if not ClassificationProvider().has_model():
        return 'Model not trained yet', 400
    predicted_label = ClassificationProvider().get_model().predict([data['text']])[0]
    return jsonify({'label': predicted_label})


@app.route("/predict_proba", methods=['POST'])
def predict_proba():
    """
    Predict probabilities for top n class labels. Requires a previous train-call
    ---
    parameters:
      - in: body
        name: body
        schema:
          required:
            - text
            - n
          properties:
            text:
              type: string
            n:
              type: integer
    responses:
      400:
        description: Invalid Text or model not previously trained
      200:
        description: Descending sorted probabilities for class labels
        schema:
          required:
            - labels
          properties:
            labels:
              type: array
              items:
                type: object
                properties:
                  label:
                    type: string
                  probability:
                    type: number
    """
    data = request.get_json()
    if data is None or 'text' not in data:
        return 'Missing text in body', 400
    if not ClassificationProvider().has_model():
        return 'Model not trained yet', 400
    p_labels = ClassificationProvider().get_model().predict_proba([data['text']], data.get('n'))[0]
    return jsonify([{'label': p[0], 'probability': float(p[1])} for p in p_labels])


@app.route("/m/predict_proba", methods=['POST'])
def predict_proba_batch():
    """
    Predict probabilities for top n class labels for all supplied texts. Requires a previous train-call
    ---
    parameters:
      - in: body
        name: body
        schema:
          required:
            - texts
            - n
          properties:
            texts:
              type: array
              items:
                type: string
            n:
              type: integer
    responses:
      400:
        description: Invalid Text or model not previously trained
      200:
        description: Descending sorted probabilities for class labels
        schema:
          required:
            - results
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                  labels:
                    type: array
                    items:
                      type: object
                      properties:
                        label:
                          type: string
                        probability:
                          type: number
    """
    data = request.get_json()
    if data is None or 'texts' not in data:
        return 'Missing texts in body', 400
    if not ClassificationProvider().has_model():
        return 'Model not trained yet', 400
    p_labels = ClassificationProvider().get_model().predict_proba(data['texts'], data.get('n'))
    res = []
    for text, labels in zip(data['texts'], p_labels):
        text_labels = [{'label': p[0], 'probability': float(p[1])} for p in labels]
        res.append({'text': text, 'labels': text_labels})

    return jsonify(res)

@app.route("/download")
def download_model():
    """
    Retrieves an implementation-specific model dump (e.g. joblib for sklearn)
    ---
    responses:
      400:
        description: Model not previously trained
      200:
        description: Dumped model as file
        content:
          application/octet-stream:
            schema:
              type: string
              format: binary
    """
    if not ClassificationProvider().has_model():
        return 'Model not trained yet', 400
    tmp_file_path = ClassificationProvider().get_model().get_dumped_model_path()
    return send_file(tmp_file_path)


@app.route('/training/status')
def get_training_status():
    """
    Checks if a trained model is available (and fully trained)
    ---
    responses:
      200:
        description: Indicator for availability of a trained model
        schema:
          properties:
            isTrained:
              type: boolean
    """
    is_trained = ClassificationProvider().has_model() and ClassificationProvider().get_model().is_trained()
    return jsonify({'is_trained': is_trained})


@app.route('/')
def entry():
    return redirect('/api/docs')


@app.route("/spec")
def spec():
    base_path = request.headers.environ.get('HTTP_SWAGGER_PREFIX') # workaround for reverse-proxy
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Text Classification Service"

    # fix paths - probably also works with undocumented flask settings
    if base_path:
        endpoints = list(swag['paths'].keys())
        for endpoint in endpoints:
            swag['paths'][base_path + endpoint[1:]] = swag['paths'].pop(endpoint)
    return jsonify(swag)


swaggerui_blueprint = get_swaggerui_blueprint(
    '/api/docs',
    '/spec',
    config={
        'app_name': "Text Classification Service"
    }
)
app.register_blueprint(swaggerui_blueprint)


if __name__ == '__main__':
    app.run()
