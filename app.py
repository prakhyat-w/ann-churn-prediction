import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = tf.keras.models.load_model('model.h5')

with open('label_ecoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

with open('ohe_geography','rb') as file:
    ohe_geography = pickle.load(file)


# Streamlit

st.title('Customer Churn Prediction')

geography = st.segmented_control("Geography", ohe_geography.categories_[0],default= 'Germany')
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
salary = st.number_input("Estimated Salary")
tenure = st.slider('Tenure',0,10)
number_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore': credit_score,
    'Gender' : label_encoder_gender.transform([gender]),
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': number_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': salary
    })

geo_encoded = ohe_geography.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded,columns= ohe_geography.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prob = prediction[0][0]

if prob > 0.5:
    st.badge("Customer is likely to churn!", icon=":material/emergency_home:", color='red')
else:
    st.badge("Customer is not likely to churn!", icon=":material/check:", color='blue')

st.write(f"Probability : {prob:.2f}")