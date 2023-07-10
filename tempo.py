import numpy as np
import pickle 

# loading the saved model

loaded_model=pickle.load(open('C:/Users/manoj/Desktop/Machine Learning/trained_model.sav','rb'))
input_data = np.array([2,197,70,45,543,30.5,0.158,53])
input_data = input_data.reshape(1,-1)
standerdized_input_data = np.asarray(input_data)
prediction = loaded_model.predict(standerdized_input_data)
if(prediction[0] == 0):
  print("Person is not diabetic")
else:
  print("The Person is diabetic")
