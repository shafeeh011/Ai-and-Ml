import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st


def diabetes_prediction(input_data):

    # load the saved model
    loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))
    X = pickle.load(open('diabetes_X.sav', 'rb'))

    # Making a Predictive System
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the data as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    # in this we need to standardize the input data, because we have standardized the training data initially
    scaler = StandardScaler()



    scaler.fit(X)
    std_data = scaler.transform(input_data_reshaped)
    print(std_data)

    # prediction
    prediction = loaded_model.predict(std_data)
    print(prediction)
    if (prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    
    # giving the title
    st.title('Diabetes Prediction Web App')
    
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Number of Glucose')
    BloodPressure = st.text_input('Number of BloodPressure')
    SkinThickness = st.text_input('Number of SkinThickness')
    BMI = st.text_input('Number of BMI')
    Insulin = st.text_input('Number of Insulin')
    DiabetesPedigreeFunction = st.text_input('Number of DiabetesPedigreeFunction')
    Age = st.text_input('Number of Age')
    
    
    
    # code for prediction
    diagnosis = ''

    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
            
            