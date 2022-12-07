import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("selected_feature_tf-idf.sav", "rb"))))


st.title('Prediksi Klasifikasi SMS')

clean_teks = st.text_input('Masukkan Sentimen')


fraud_detection = ''

if st.button('Prediksi SMS'):
    predict_fraud  = model_fraud.predict(loaded_vec.fit_transform([clean_teks]))

    if (predict_fraud == 0):
        fraud_detection = 'SMS Normal'
    elif (predict_fraud == 1):
        fraud_detection = 'SMS Fraud'
    else: 
        fraud_detection = 'SMS Promo'

st.success(fraud_detection)