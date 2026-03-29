import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load('model.pkl')
encoder_genre = joblib.load('encoder_genre.pkl')
encoder_publisher = joblib.load('encoder_publisher.pkl')
encoder_language = joblib.load('encoder_language.pkl')

st.title("Fantasy Book Popularity Predictor")
st.markdown(
    "Enter the details of a book below to predict how many Goodreads ratings it is likely to receive."
)

st.header("Book Details")

col1, col2 = st.columns(2)

with col1:
    num_pages = st.number_input("Number of Pages", min_value=1, max_value=5000, value=350)
    year = st.number_input("Publication Year", min_value=1900, max_value=2025, value=2010)
    genre = encoder_genre.transform(["Fantasy"])[0]

with col2:
    language = st.selectbox("Language", sorted(encoder_language.classes_))
    publisher = st.selectbox("Publisher", sorted(encoder_publisher.classes_))

if st.button("Predict Popularity"):
    genre_enc = genre
    publisher_enc = encoder_publisher.transform([publisher])[0]
    language_enc = encoder_language.transform([language])[0]

    X = pd.DataFrame([[num_pages, year, genre_enc, publisher_enc, language_enc]],
                     columns=['num_pages', 'year', 'genre', 'publisher', 'language_code'])

    log_pred = model.predict(X)[0]
    pred = int(np.expm1(log_pred))

    st.markdown("---")
    st.subheader("Predicted Ratings Count")
    st.metric(label="Estimated number of Goodreads ratings", value=f"{pred:,}")

    if pred < 1000:
        st.info("This book is predicted to have a niche audience.")
    elif pred < 10000:
        st.success("This book is predicted to have a moderate audience.")
    elif pred < 100000:
        st.success("This book is predicted to be quite popular!")
    else:
        st.balloons()
        st.success("This book is predicted to be a bestseller!")

st.markdown("---")
st.caption("Model: Random Forest Regressor trained on 11,127 Goodreads books. Target was log-transformed before training.")