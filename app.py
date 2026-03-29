import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


@st.cache_resource
def train_model():
    books = pd.read_csv("data/Goodreads_books_with_genres.csv")

    books["genres_split"] = books["genres"].str.split(';')
    books_exploded = books.explode("genres_split").rename(columns={"genres_split": "genre"})
    books_exploded["genre"] = books_exploded["genre"].str.strip()
    books_exploded = books_exploded.dropna(subset=["genre"])
    books_exploded = books_exploded[books_exploded["genre"] != ""]
    books_exploded["genre"] = books_exploded["genre"].str.split(",")
    books_exploded1 = books_exploded.explode("genre")
    books_exploded1["genre"] = books_exploded1["genre"].str.strip()

    genre_counts = books_exploded1.groupby("genre")["Book Id"].nunique()
    common_genres = genre_counts[genre_counts >= 500].index
    books_filtered = books_exploded1[books_exploded1["genre"].isin(common_genres)].copy()

    genre_map = {
        'Adult Fiction': 'Fiction', 'Science Fiction Fantasy': 'Fantasy',
        'Science Fiction': 'Fiction', 'Historical Fiction': 'Fiction',
        'Literary Fiction': 'Fiction', 'Mystery Thriller': 'Thriller'
    }
    books_filtered["genre"] = books_filtered["genre"].apply(lambda x: genre_map.get(x, x))
    books_filtered = books_filtered.drop_duplicates(subset=["Book Id", "genre"])

    books_filtered['year'] = pd.to_datetime(
        books_filtered['publication_date'], format='%m/%d/%Y', errors='coerce'
    ).dt.year
    books_filtered = books_filtered.dropna(subset=['year']).copy()
    books_filtered['year'] = books_filtered['year'].astype(int)

    X = books_filtered[['num_pages', 'year', 'genre', 'publisher', 'language_code']].copy()
    y = np.log1p(books_filtered['ratings_count'])

    enc_genre = LabelEncoder()
    enc_pub = LabelEncoder()
    enc_lang = LabelEncoder()
    X['genre'] = enc_genre.fit_transform(X['genre'])
    X['publisher'] = enc_pub.fit_transform(X['publisher'])
    X['language_code'] = enc_lang.fit_transform(X['language_code'])

    model = RandomForestRegressor(
        n_estimators=200, max_features=3, min_samples_leaf=2,
        max_depth=None, max_leaf_nodes=None, random_state=44, n_jobs=-1
    )
    model.fit(X, y)

    return model, enc_genre, enc_pub, enc_lang


with st.spinner("Loading model (this may take a minute on first run)..."):
    model, encoder_genre, encoder_publisher, encoder_language = train_model()

st.title("Fantasy Book Popularity Predictor")
st.markdown(
    "Enter the details of a book below to predict how many Goodreads ratings it is likely to receive."
)

st.header("Book Details")

col1, col2 = st.columns(2)

with col1:
    num_pages = st.number_input("Number of Pages", min_value=1, max_value=5000, value=350)
    year = st.number_input("Publication Year", min_value=1900, max_value=2025, value=2010)

with col2:
    language = st.selectbox("Language", sorted(encoder_language.classes_))
    publisher = st.selectbox("Publisher", sorted(encoder_publisher.classes_))

genre = encoder_genre.transform(["Fantasy"])[0]

if st.button("Predict Popularity"):
    publisher_enc = encoder_publisher.transform([publisher])[0]
    language_enc = encoder_language.transform([language])[0]

    X_input = pd.DataFrame(
        [[num_pages, year, genre, publisher_enc, language_enc]],
        columns=['num_pages', 'year', 'genre', 'publisher', 'language_code']
    )

    pred = int(np.expm1(model.predict(X_input)[0]))

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