Movie Recommendation Engine

Collaborative filtering recommender on the Netflix Prize dataset, using SVD from the surprise library. Give it a customer ID, it predicts what they'd rate other movies and shows the top 5.

How it works
Loads the ratings, drops movies/customers below the 60th percentile of rating count (cuts out data too sparse to learn from)
Trains SVD on a 100k-row subset — full dataset is 24M+ rows and just isn't practical to train locally
For a given customer, predicts a rating on every movie they haven't rated yet and returns the top 5
Getting the data

Netflix Prize dataset from Kaggle (not included here, it's a few GB):

https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

Need combined_data_1.txt and movie_titles.csv, both go in a data/ folder next to app.py:

data/
  combined_data_1.txt
  movie_titles.csv
Running it
bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app.py

Click "Train model" on first load.

MIT
