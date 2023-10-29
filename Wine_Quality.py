import pickle
import streamlit as st

model = pickle.load(open('Estimasi_Wine_Quality.sav','rb'))

st.title('Estimasi Wine Quality')
residual_sugar   = st.number_input('input residual sugar')
pH  = st.number_input('input pH')
alcohol = st.number_input('input alcohol')
chlorides   = st.number_input('input chlorides')
sulphates   = st.number_input('input sulphates')
fixed_acidity   = st.number_input('input fixed acidity')
volatile_acidity = st.number_input('input volatile acidity')
citric_acid = st.number_input('input citric acid')
free_sulfur_dioxide = st.number_input('input free sulfur dioxide')
total_sulfur_dioxide = st.number_input('input total sulfur dioxide')
density = st.number_input('input density')

predict = ''

if st.button('Cek Kualitas Wine'):
    predict = model.predict(
        [[residual_sugar, pH, alcohol, chlorides, sulphates, fixed_acidity, volatile_acidity, citric_acid, free_sulfur_dioxide, total_sulfur_dioxide, density]]
    )
    st.write('Estimasi Quality Wine:', predict)
