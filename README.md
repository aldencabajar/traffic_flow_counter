# Traffic flow counter :vertical_traffic_light:

## Introduction
Hello everyone! The following code will be a current work-in-progress app for traffic flow counting. My hope is to make this something of value for my city to be used for traffic flow management.This uses the `yolo-v3` computer vision model to vehicles.

## Setup
To run the app locally, install the necessary python packages by: 

```python
pip install -r requirements.txt
```
For better reproducibility, make sure to open up a python environment using `virtualenv` or any of your favorite python environment packages. 

The app contains a custom slider. To use it, make sure you have `npm` installed and then run the following commands.
```bash
cd components/custom_slider/frontend/
npm install 
```
Afterwards, run `npm run build`

This should recreate the build package for the custom slider. 

pull the necessary models and data needed using
```
bash dependencies.sh
```

Run the app!
```
streamlit run app/streamlit-app.py
```

## Future Developments
Here are some of my ideas that would be neat additions to the API: 
+ Speed measurement
+ Vehicle distribution measurement
+ Traffic density


