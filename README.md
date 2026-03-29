# Book Popularity Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fantasy-book-analyzer.streamlit.app/)

A data science project that analyzes Goodreads book data to uncover what makes a book popular, and builds a machine learning model to predict book popularity across genres.

🚀 **[Try the live app here](https://fantasy-book-analyzer.streamlit.app/)**

## Project Overview

Using a dataset of **11,127 books** from [Kaggle](https://www.kaggle.com/datasets/middlelight/goodreadsbookswithgenres), this project explores book popularity through exploratory data analysis (EDA) and predictive modeling, with a focus on the fantasy genre during EDA.

**Key questions explored:**
- Are specific sub-genres (epic, dark, romance) rated higher?
- Does publication year affect ratings?
- Do longer books get higher ratings?
- Is there a relationship between genre popularity and average ratings?

## Key Findings

- **Genre popularity negatively correlates with ratings** (Spearman rho = -0.47, p = 0.001) — niche genres like Graphic Novels score higher due to smaller, more dedicated audiences.
- **Book length and publisher are the strongest predictors** of popularity (35.3% and 32.9% feature importance respectively).
- **Ratings are left-skewed**, peaking around 4.0, suggesting Goodreads users tend to rate books they enjoyed.
- **Language has minimal impact on ratings** — average ratings across 27 languages fall in a narrow 3.9-4.5 band.
- **Popularity and quality are largely independent** — `average_rating` is barely correlated with any numeric feature.

## Machine Learning Model

A **Random Forest Regressor** was trained to predict `ratings_count` (a proxy for popularity) across all genres. The target was log-transformed (`log1p`) to reduce the influence of extreme outliers before training.

| Feature | Importance |
|---|---|
| Number of pages | 35.3% |
| Publisher | 32.9% |
| Publication year | 16.5% |
| Language | 11.9% |
| Genre | 3.5% |

**Model performance:**
- Single train/test split R² = 0.9193 (log scale)
- Nested cross-validation R² = 0.7806 ± 0.0059 (log scale) — a more reliable generalisation estimate

A baseline linear regression achieved R² = 0.0705, confirming the non-linear nature of the problem.

## Deployment

The model is deployed as an interactive web app on Streamlit Community Cloud.

👉 [https://fantasy-book-analyzer.streamlit.app/](https://fantasy-book-analyzer.streamlit.app/)

The app trains the model at startup directly from the dataset — no pre-saved model files required. Users can select a genre, enter book details (pages, year, publisher, language) and receive an instant popularity prediction.

## Project Structure

```
fantasy-book-analyzer/
├── Book popularity predictor.ipynb   # Main analysis notebook
├── app.py                            # Streamlit web app
├── data/
│   └── Goodreads_books_with_genres.csv
├── requirements.txt
└── README.md
```

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook "Book popularity predictor.ipynb"
   ```
4. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Dataset

Source: [Goodreads Books with Genres](https://www.kaggle.com/datasets/middlelight/goodreadsbookswithgenres) on Kaggle.

- 11,127 books, 13 columns
- Features: title, author, average rating, ISBN, language, page count, ratings count, publication date, publisher, genres
- 6,643 unique authors across 27 languages and 2,292 publishers