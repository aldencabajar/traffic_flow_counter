FROM heroku/heroku:20

# RUN apt update \ 
#     && apt install -y sudo wget 
# RUN sudo apt-get update
RUN apt update
RUN apt install -y python3-pip 
RUN apt install -y npm

ENV APP_HOME /app/
WORKDIR ${APP_HOME}


COPY requirements.txt . 
RUN pip3 install -r requirements.txt

COPY . .
RUN wget https://pjreddie.com/media/files/yolov3.weights
RUN mv yolov3.weights data/

# prepare building of component 
WORKDIR ${APP_HOME}/components/custom_slider/frontend/
RUN npm install
RUN npm run build

EXPOSE 8501

WORKDIR ${APP_HOME}

CMD sh setup.sh && streamlit run app/streamlit-app.py











