import streamlit as st
import pickle
import numpy as np

def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

st.title("Diabetes Prediction App")

st.write("""
            This app allows you to make predictions on whether a woman has diabetes or not.
         Please enter the values for the features below to get a prediction.
         """)

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=800, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input("Age (Years)", min_value=0, max_value=120, value=0)

models = ["KNN Algorithm", "Logistic Regression Algorithm", "Random Forest Algorithm", "AdaBoost Algorithm"]
model_selection = st.selectbox("Select the Algorithm", models)

input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)

if model_selection == "KNN Algorithm":
    model = load_model('knn_grid_search.pkl')
elif model_selection == "Logistic Regression Algorithm":
    model = load_model('log_reg_grid_search.pkl')
elif model_selection == "Random Forest Algorithm":
    model = load_model('rand_frst_grid_search.pkl')
elif model_selection == "AdaBoost Algorithm":
    model = load_model('ada_bst_grid_search.pkl')

if st.button("Predict"):
    prediction = model.predict(input_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("Prediction: The woman is likely to have diabetes (Outcome: 1).")
    else:
        st.write("Prediction: The woman is unlikely to have diabetes (Outcome: 0).")