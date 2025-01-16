import json
import requests

url = 'http://127.0.0.1:8000/diabetes_prediction'
#1,103,30,38,83,43.3,0.183,33
input_data = {
    
    'Pregnancies' : 1,
    'Glucose' : 103,
    'BloodPressure' : 30,
    'SkinThickness' : 38,
    'Insulin' : 83,
    'BMI' : 43.3,
    'DiabetesPedigreeFunction' : 0.183,
    'Age' : 33
}

# convert the input dict to .json file

input_json = json.dumps(input_data)
response = requests.post(url,data=input_json)
print(response.text)

