FROM python:3.6-slim-stretch

RUN apt update \ 
    && apt install -y wget  
ENV APP_HOME /app/
WORKDIR ${APP_HOME}


COPY requirements.txt . 
RUN pip3 install -r requirements.txt

COPY . .
RUN wget https://pjreddie.com/media/files/yolov3.weights
RUN mv yolov3.weights data/







