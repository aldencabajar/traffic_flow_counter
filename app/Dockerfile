FROM python:3.6-slim-stretch

RUN apt update \ 
    && apt install -y wget  

ENV APP_HOME /app/
WORKDIR ${APP_HOME}/app/

RUN wget https://pjreddie.com/media/files/yolov3.weights

COPY requirements.txt . 
RUN pip3 install -r requirements.txt

COPY . .
WORKDIR ${APP_HOME}