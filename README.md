# zimp-clf-service
RESTful classification service used for experiments in the zimp pipeline

## Supported Models
Pass desired model as parameter of the train POST-request
* Sklearn SVM
* Sklearn Decision Tree
* Sklearn Random Forest
* FastText
* Huggingface DistillBert
* Huggingface GermanBert

## Startup
1. `pip install -r requirements.txt`
2. `CUDA_VISIBLE_DEVICES=0 & python -m wsgi` or `gunicorn --bind 0.0.0.0:5000 wsgi:app`
3. Goto 127.0.0.1:5000 and check the api reference

## API
![API Reference](doc/api_ref.png)

## Docker Setup
see https://hub.docker.com/repository/docker/freecraver/zimp-clf-service
1. (optional) `docker build --tag zimp-clf-service .`
2. or just `docker container run -p 5000:5000 freecraver/zimp-clf-service`
3. Goto 127.0.0.1:5000 and check the api reference

