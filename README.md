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

Click "Train model" on first load. It's cached after that so switching customers doesn't retrain from scratch.

If pip install fails on scikit-surprise

This is the most common thing that breaks here, and it's not really a bug in the code — scikit-surprise compiles some Cython/C extensions during install, and that needs a C++ build toolchain that a lot of machines don't have set up by default.

Windows: install "Microsoft C++ Build Tools" first (visualstudio.microsoft.com/visual-cpp-build-tools), then retry pip install scikit-surprise.

Mac: run xcode-select --install first if you haven't already, then retry.

Easiest fix on any OS: use conda instead of pip —

bash
conda install -c conda-forge scikit-surprise

conda ships prebuilt binaries so it skips the compile step entirely.

If it's still failing after that, the actual error message from pip usually says exactly what's missing — worth reading past the giant wall of red text, the real reason is often in the last few lines.

Limitations
Only trained on a subset, so predictions are okay but not as strong as training on the full 24M ratings would give
No cold-start handling — can only recommend for customers already in the training data
RMSE is shown in the app instead of just hiding it, so you can actually see how good the model is instead of taking it on faith
Stack
surprise for SVD + cross-validation
pandas for data wrangling
Streamlit for the UI
License

MIT
