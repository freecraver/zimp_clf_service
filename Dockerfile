FROM nvidia/cuda:11.5.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
&& apt-get install build-essential -y \
&& apt-get install python3 python3-pip -y \
 && apt-get clean

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "wsgi:app", "-w", "1", "--threads", "1", "--timeout", "0", "-b", "0.0.0.0:5000"]

#CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]