mkdir -p ~/.streamlit/
echo "[server]\n\
headless=true\n\
port=$PORT\n\
enableCORS=false\n\
">~/.streamlit/config.toml

wget https://pjreddie.com/media/files/yolov3-tiny.weights
mv yolov3-tiny.weights data/
