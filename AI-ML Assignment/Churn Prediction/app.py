import streamlit as st
import pickle
import numpy as np

def load_model(model_file) :
    with open(model_file, 'rb') as file :
        model = pickle.load(file)

    return model

st.title("Churn Prediction App")

st.write("""
            This app allows you to make predictions using the trained model.
         Please enter the values for the features below to get a prediction.
         """)

gender = st.selectbox("Gender", options=["Female", "Male"], help="Choose the gender of the customer")
gender_map = {"Female": 0.00, "Male": 1.00}
gender = gender_map[gender]

senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"], help="Is the customer a senior citizen?")
senior_citizen_map = {"No": 0.00, "Yes": 1.00}
senior_citizen = senior_citizen_map[senior_citizen]

partner = st.selectbox("Partner", options=["No", "Yes"], help="Does the customer have a partner?")
partner_map = {"No": 0.00, "Yes": 1.00}
partner = partner_map[partner]

dependents = st.selectbox("Dependents", options=["No", "Yes"], help="Does the customer have dependents?")
dependents_map = {"No": 0.00, "Yes": 1.00}
dependents = dependents_map[dependents]

tenure = st.number_input("Tenure (Months)", value=0.00)

phone_service = st.selectbox("Phone Service", options=["No", "Yes"], help="Does the customer have phone service?")
phone_service_map = {"No": 0.00, "Yes": 1.00}
phone_service = phone_service_map[phone_service]

multiple_lines = st.selectbox("Multiple Lines", options=["No", "No phone service", "Yes"], help="Does the customer have multiple phone lines?")
multiple_lines_map = {"No": 0.00, "No phone service":1.00, "Yes": 2.00}
multiple_lines = multiple_lines_map[multiple_lines]

internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No internet"], help="What kind of internet service does the customer have?")
internet_service_map = {"DSL": 0.00, "Fiber optic": 1.00, "No internet": 2.00}
internet_service = internet_service_map[internet_service]

online_security = st.selectbox("Online Security", options=["No", "No internet service", "Yes"], help="Does the customer have online security?")
online_security_map = {"No": 0.00, "No internet service":1.00, "Yes": 2.00}
online_security = online_security_map[online_security]

online_backup = st.selectbox("Online Backup", options=["No", "No internet service", "Yes"], help="Does the customer have online backup?")
online_backup_map = {"No": 0.00, "No internet service":1.00, "Yes": 2.00}
online_backup = online_backup_map[online_backup]

device_protection = st.selectbox("Device Protection", options=["No", "No internet service", "Yes"], help="Does the customer have device protection?")
device_protection_map = {"No": 0.00, "No internet service":1.00, "Yes": 2.00}
device_protection = device_protection_map[device_protection]

tech_support = st.selectbox("Tech Support", options=["No", "No internet service", "Yes"], help="Does the customer have tech support?")
tech_support_map = {"No": 0.00, "No internet service":1.00, "Yes": 2.00}
tech_support = tech_support_map[tech_support]

streaming_tv = st.selectbox("Streaming TV", options=["No", "No internet service", "Yes"], help="Does the customer use streaming TV?")
streaming_tv_map = {"No": 0.00, "No internet service":1.00, "Yes": 2.00}
streaming_tv = streaming_tv_map[streaming_tv]

streaming_movies = st.selectbox("Streaming Movies", options=["No", "No internet service", "Yes"], help="Does the customer use streaming movies?")
streaming_movies_map = {"No": 0.00, "No internet service":1.00, "Yes": 2.00}
streaming_movies = streaming_movies_map[streaming_movies]

contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], help="What kind of contract does the customer have?")
contract_map = {"Month-to-month": 0.00, "One year": 1.00, "Two year": 2.00}
contract = contract_map[contract]

paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"], help="Does the customer have paperless billing?")
paperless_billing_map = {"No": 0.00, "Yes": 1.00}
paperless_billing = paperless_billing_map[paperless_billing]

payment_method = st.selectbox("Payment Method", options=["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], help="How does the customer pay?")
payment_method_map = {"Bank transfer (automatic)": 0.00, "Credit card (automatic)": 1.00, "Electronic check": 2.00, "Mailed check": 3.00}
payment_method = payment_method_map[payment_method]

monthly_charges = st.number_input("Monthly Charges", value=0.00)

total_charges = st.number_input("Total Charges", value=0.00)

input_data = np.array([gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, 
                       internet_service, online_security, online_backup, device_protection, tech_support, 
                       streaming_tv, streaming_movies, contract, paperless_billing, payment_method, 
                       monthly_charges, total_charges]).reshape(1, -1)

models = ["KNN Algorithm", "Logistic Regression Algorithm", "Random Forest Algorithm", "AdaBoost Algorithm"]

model_selection = st.selectbox("Select the Algorithm", models)

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
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")