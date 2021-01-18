FROM python:3.6-slim-stretch

ENV APP_HOME /app/
WORKDIR ${APP_HOME}


COPY requirements.txt . 
RUN pip3 install -r requirements.txt

COPY . .
RUN make get_yolo_v3_weights 



