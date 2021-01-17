FROM heroku/heroku:20

# install pip
RUN apt-get update && apt-get install -y python3-pip

ENV APP_HOME /app/
WORKDIR ${APP_HOME}


COPY requirements.txt . 
RUN pip3 install -r requirements.txt

COPY . .
RUN make all



