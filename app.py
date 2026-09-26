import streamlit as st
import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

st.set_page_config(page_title="Movie Recommendation Engine", page_icon="🎬", layout="wide")

DATA_DIR = "data"
RATINGS_FILE = os.path.join(DATA_DIR, "combined_data_1.txt")
TITLES_FILE = os.path.join(DATA_DIR, "movie_titles.csv")

# full dataset is 24 million+ rows, way too slow to train on locally.
# 100k is a decent tradeoff between "actually finishes" and "model isn't garbage"
ROWS_FOR_TRAINING = 100_000


@st.cache_data
def load_ratings():
    df = pd.read_csv(RATINGS_FILE, header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

    # raw file has "12345:" as its own row marking a new movie id, instead of
    # a real column. forward-filling it down onto the rating rows below it
    movie_id = None
    movie_ids = []
    for cust_id in df['Cust_Id']:
        if ':' in cust_id:
            movie_id = int(cust_id.replace(':', ''))
        movie_ids.append(movie_id)
    df['Movie_Id'] = movie_ids

    df = df[df['Rating'].notna()].copy()
    df['Cust_Id'] = df['Cust_Id'].astype(int)
    df['Rating'] = df['Rating'].astype(float)
    return df


@st.cache_data
def load_titles():
    return pd.read_csv(
        TITLES_FILE, encoding='ISO-8859-1', header=None,
        usecols=[0, 1, 2], names=['Movie_Id', 'Year', 'Name'],
    )


def filter_sparse(df, quantile=0.6):
    # dropping movies/customers below the 60th percentile of rating count —
    # same cutoff used in the original notebook, seemed to work fine
    movie_counts = df.groupby('Movie_Id')['Rating'].count()
    drop_movies = movie_counts[movie_counts < movie_counts.quantile(quantile)].index

    cust_counts = df.groupby('Cust_Id')['Rating'].count()
    drop_custs = cust_counts[cust_counts < cust_counts.quantile(quantile)].index

    df = df[~df['Movie_Id'].isin(drop_movies)]
    df = df[~df['Cust_Id'].isin(drop_custs)]
    return df, drop_movies


@st.cache_resource
def train_model(ratings_df):
    reader = Reader()
    subset = ratings_df[['Cust_Id', 'Movie_Id', 'Rating']][:ROWS_FOR_TRAINING]
    data = Dataset.load_from_df(subset, reader)
    model = SVD()
    scores = cross_validate(model, data, measures=['RMSE'], cv=3, verbose=False)
    # cross_validate doesn't leave us a fitted model on the full data, so fit
    # one more time on everything before actually using it for predictions
    trainset = data.build_full_trainset()
    model.fit(trainset)
    return model, scores['test_rmse'].mean()


def recommend_for_user(model, titles_df, drop_movies, cust_id, n=5):
    candidates = titles_df[~titles_df['Movie_Id'].isin(drop_movies)].copy()
    candidates['Estimate_Score'] = candidates['Movie_Id'].apply(lambda mid: model.predict(cust_id, mid).est)
    return candidates.sort_values('Estimate_Score', ascending=False).head(n)


def main():
    st.title('🎬 Movie Recommendation Engine')
    st.caption('SVD collaborative filtering on the Netflix Prize dataset')

    if not os.path.exists(RATINGS_FILE) or not os.path.exists(TITLES_FILE):
        st.error(f"couldn't find the data files — need {RATINGS_FILE} and {TITLES_FILE}, check the README")
        return

    with st.spinner('loading ratings...'):
        ratings_df = load_ratings()
        titles_df = load_titles()

    st.write(f"{len(ratings_df):,} ratings, {ratings_df['Movie_Id'].nunique():,} movies, "
             f"{ratings_df['Cust_Id'].nunique():,} customers")

    filtered_df, drop_movies = filter_sparse(ratings_df)

    if 'model' not in st.session_state:
        if st.button('Train model', type='primary'):
            with st.spinner(f'training on {ROWS_FOR_TRAINING:,} ratings, gimme a minute...'):
                model, rmse = train_model(filtered_df)
                st.session_state.model = model
                st.session_state.rmse = rmse
            st.rerun()
        else:
            st.info('hit train first — only need to do this once per session')
            return

    st.success(f'model trained (RMSE: {st.session_state.rmse:.3f})')
    st.divider()

    known_users = sorted(filtered_df['Cust_Id'].unique())
    cust_id = st.selectbox('Customer ID', known_users[:200])  # capping the dropdown, thousands of ids gets slow to render

    user_ratings = filtered_df[filtered_df['Cust_Id'] == cust_id]
    st.write(f"this customer rated {user_ratings['Movie_Id'].nunique()} movies")

    if st.button('Get recommendations'):
        top5 = recommend_for_user(st.session_state.model, titles_df, drop_movies, cust_id)
        st.subheader('Top 5')
        st.dataframe(top5[['Name', 'Year', 'Estimate_Score']], use_container_width=True)


if __name__ == '__main__':
    main()
