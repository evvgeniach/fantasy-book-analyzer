import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Book Popularity Predictor",
    page_icon="📚",
    layout="centered"
)

LANG_MAP = {
    'ale':   'Aleut',
    'ara':   'Arabic',
    'en-CA': 'English',
    'en-GB': 'English',
    'en-US': 'English',
    'eng':   'English',
    'enm':   'Middle English',
    'fre':   'French',
    'ger':   'German',
    'gla':   'Scottish Gaelic',
    'glg':   'Galician',
    'grc':   'Greek',
    'ita':   'Italian',
    'jpn':   'Japanese',
    'lat':   'Latin',
    'msa':   'Malay',
    'mul':   'Multiple Languages',
    'nl':    'Dutch',
    'nor':   'Norwegian',
    'por':   'Portuguese',
    'rus':   'Russian',
    'spa':   'Spanish',
    'srp':   'Serbian',
    'swe':   'Swedish',
    'tur':   'Turkish',
    'wel':   'Welsh',
    'zho':   'Chinese',
}

GENRE_MAP = {
    'Adult Fiction': 'Fiction',
    'Science Fiction Fantasy': 'Fantasy',
    'Science Fiction': 'Fiction',
    'Historical Fiction': 'Fiction',
    'Literary Fiction': 'Fiction',
    'Mystery Thriller': 'Thriller',
}


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

    books_filtered["genre"] = books_filtered["genre"].apply(lambda x: GENRE_MAP.get(x, x))
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


with st.spinner("Training model on first run, please wait..."):
    model, encoder_genre, encoder_publisher, encoder_language = train_model()

# Build readable genre options (deduplicated after genre_map consolidation)
raw_genres = sorted(encoder_genre.classes_)
genre_display = sorted(set(GENRE_MAP.get(g, g) for g in raw_genres))

# Build readable language options (label -> one representative code)
lang_options = {}
for code in sorted(encoder_language.classes_):
    label = LANG_MAP.get(code, code)
    if label not in lang_options:
        lang_options[label] = code
lang_display = sorted(lang_options.keys())

# --- Header ---
st.markdown("# 📚 Book Popularity Predictor")
st.markdown(
    "Predict how many Goodreads ratings a book is likely to receive, "
    "based on its genre, length, publisher, publication year, and language."
)
st.markdown("---")

# --- Input form ---
st.markdown("### 📝 Book Details")

col1, col2 = st.columns(2)

with col1:
    genre_label = st.selectbox("📖 Genre", genre_display,
                               index=genre_display.index("Fantasy") if "Fantasy" in genre_display else 0)
    num_pages = st.number_input("📄 Number of Pages", min_value=1, max_value=5000, value=350,
                                help="Total number of pages in the book")
    year = st.number_input("📅 Publication Year", min_value=1900, max_value=2025, value=2010,
                           help="Year the book was first published")

with col2:
    lang_label = st.selectbox("🌍 Language", lang_display,
                              index=lang_display.index("English") if "English" in lang_display else 0)
    publisher = st.selectbox("🏢 Publisher", sorted(encoder_publisher.classes_),
                             help="Select the book's publisher")

st.markdown("")

# --- Predict ---
if st.button("🔮 Predict Popularity", use_container_width=True):

    # Encode genre — use the raw label that the encoder knows
    # If the selected display genre was consolidated, find its encoder class
    genre_enc_label = genre_label
    if genre_enc_label not in encoder_genre.classes_:
        # Fall back to first matching class after reverse-mapping
        for raw in encoder_genre.classes_:
            if GENRE_MAP.get(raw, raw) == genre_label:
                genre_enc_label = raw
                break
    genre_enc = encoder_genre.transform([genre_enc_label])[0]

    lang_code = lang_options[lang_label]
    language_enc = encoder_language.transform([lang_code])[0]
    publisher_enc = encoder_publisher.transform([publisher])[0]

    X_input = pd.DataFrame(
        [[num_pages, year, genre_enc, publisher_enc, language_enc]],
        columns=['num_pages', 'year', 'genre', 'publisher', 'language_code']
    )

    pred = int(np.expm1(model.predict(X_input)[0]))

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")
    st.metric(label="Estimated Goodreads Ratings", value=f"{pred:,}")

    if pred < 1000:
        st.info("🔍 Niche audience — likely to appeal to dedicated fans of the genre.")
    elif pred < 10000:
        st.success("📖 Moderate audience — solid readership expected.")
    elif pred < 100000:
        st.success("⭐ Popular book — strong readership predicted!")
    else:
        st.balloons()
        st.success("🏆 Bestseller territory — exceptional popularity predicted!")

# --- About section ---
st.markdown("---")
with st.expander("ℹ️ About this model"):
    st.markdown("""
    This app uses a **Random Forest Regressor** trained on 11,127 Goodreads books.

    **Features used:**
    - Genre
    - Number of pages
    - Publication year
    - Publisher
    - Language

    **Model performance (nested cross-validation):**
    - R² = 0.7806 ± 0.0059 (log scale)

    **Top predictors:** number of pages (35.3%) and publisher (32.9%).

    The target variable (`ratings_count`) was log-transformed before training to reduce the influence of outliers.
    Predictions are back-transformed for display.
    """)