import numpy as np
import pickle
loaded_model=pickle.load(open('C:/Users/manoj/Desktop/Machine Learning/trained_model.sav','rb'))

input_data = np.array([1,89,66,23,94,28.1,0.167,21,0])
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
if(prediction[0] == 0):
  print("Person is not diabetic")
else:
  print("The Person is diabetic")