import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('C:/Users/manoj/Desktop/Machine Learning/trained_model.sav','rb'))


# creating function for prediction

def diabetes_prediction(input_data):
    
    
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    if(prediction[0] == 0):
        return "Person is not diabetic"
    else:
        return "The Person is diabetic"
    

def main():


    # giving title
    st.title("Diabetes prediction web app")

    # getting input data from user
     #Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction Age
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('BloodPressure')
    SkinThickness=st.text_input('SkinThickness')

    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    Age=st.text_input('Age of the person')


    # code for prediction

    diagnosis=''

    #creating button for predicting value
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])


    st.success(diagnosis)



if __name__ == '__main__':
        main()
    







