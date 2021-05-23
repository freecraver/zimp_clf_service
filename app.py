"""
This is very lightweight classification service used for training and evaluating models under different scenarios
in a containerized env
"""

from flask import Flask, jsonify, request
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint

import pandas as pd

from nlp.classification_provider import ClassificationProvider

app = Flask(__name__)


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
    responses:
      400:
        description: Required file missing or having invalid format
      200:
        description: Model successfully trained
    """
    if 'file' not in request.files:
        return 'File upload required', 400
    df = pd.read_csv(request.files.get('file'))
    if df.shape[1] != 2 or len(set(df.columns) - set(['target', 'text'])) > 0:
        return 'Supplied invalid header columns. Must be text, target', 400

    ClassificationProvider().get_model().train(df['text'], df['target'])
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
    if 'text' not in data:
        return 'Missing text in body', 400
    model = ClassificationProvider().get_model()
    if not model.is_trained():
        return 'Model not trained yet', 400
    predicted_label = model.predict(data['text'])
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
    if 'text' not in data:
        return 'Missing text in body', 400
    model = ClassificationProvider().get_model()
    if not model.is_trained():
        return 'Model not trained yet', 400
    p_labels = model.predict_proba(data['text'], data.get('n'))
    return jsonify([{'label': p[0], 'probability': p[1]} for p in p_labels])


@app.route("/spec")
def spec():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Text Classification Service"
    return jsonify(swag)


swaggerui_blueprint = get_swaggerui_blueprint(
    '/api/docs',
    'http://127.0.0.1:5000/spec',
    config={
        'app_name': "Text Classification Service"
    }
)
app.register_blueprint(swaggerui_blueprint)


if __name__ == '__main__':
    app.run()
