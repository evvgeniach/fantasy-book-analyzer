# Fantasy Book Popularity Analyzer

A data science project that analyzes Goodreads book data to uncover what makes a fantasy book popular, and builds a machine learning model to predict book popularity.

## Project Overview

Using a dataset of **11,127 books** from [Kaggle](https://www.kaggle.com/datasets/middlelight/goodreadsbookswithgenres), this project explores the fantasy genre through exploratory data analysis (EDA) and predictive modeling.

**Key questions explored:**
- Are specific sub-genres (epic, dark, romance) rated higher?
- Does publication year affect ratings?
- Do longer fantasy books get higher ratings?
- Is there a relationship between genre popularity and average ratings?

## Key Findings

- **Genre popularity negatively correlates with ratings** (Spearman rho = -0.47, p = 0.001) — niche genres like Graphic Novels score higher due to smaller, more dedicated audiences.
- **Book length is the strongest predictor** of popularity (46.8% feature importance), followed by publisher (35.0%).
- **Ratings are left-skewed**, peaking around 4.0, suggesting Goodreads users tend to rate books they enjoyed.
- **Language has minimal impact** — average ratings across 27 languages fall in a narrow 3.9–4.5 band.

## Machine Learning Model

A **Random Forest Regressor** was trained to predict `ratings_count` (a proxy for popularity) using:

| Feature | Importance |
|---|---|
| Number of pages | 46.8% |
| Publisher | 35.0% |
| Publication year | 13.2% |
| Genre | 4.5% |
| Language | 0.6% |

**Model performance (nested cross-validation):**
- R² = 0.7677 ± 0.0184
- RMSE = 60,863 ± 3,876

A baseline linear regression achieved R² = 0.004, confirming the non-linear nature of the problem.

## Project Structure

```
fantasy-book-analyzer/
├── Book popularity predictor.ipynb   # Main analysis notebook
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

## Dataset

Source: [Goodreads Books with Genres](https://www.kaggle.com/datasets/middlelight/goodreadsbookswithgenres) on Kaggle.

- 11,127 books, 13 columns
- Features: title, author, average rating, ISBN, language, page count, ratings count, publication date, publisher, genres
- 6,643 unique authors across 27 languages and 2,292 publishers
