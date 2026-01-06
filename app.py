import streamlit as st
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import process

# -------------------------------
# LOAD LIGHTWEIGHT FILES ONLY
# -------------------------------
books = pd.read_csv("books_meta.csv", compression="gzip")

isbn_index = pickle.load(open("isbn_index.pkl", "rb"))

svd_data = np.load("reduced_matrix.npz")
reduced_matrix = svd_data["reduced_matrix"]

svd_knn = pickle.load(open("svd_knn.pkl", "rb"))

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📚 Book Recommendation System (SVD-Based CF)")

book_titles = books['Book-Title'].values
selected_book = st.selectbox("Select a Book", book_titles)

# -------------------------------
# RECOMMEND FUNCTION
# -------------------------------
def recommend_svd_from_title(book_title, top_n=5):

    match, score, _ = process.extractOne(book_title, book_titles)

    if score < 60:
        return None

    isbn = books.loc[
        books['Book-Title'] == match, 'ISBN'
    ].iloc[0]

    if isbn not in isbn_index:
        return None

    idx = isbn_index.index(isbn)   # or dict lookup (see note)

    distances, indices = svd_knn.kneighbors(
        reduced_matrix[idx].reshape(1, -1),
        n_neighbors=top_n + 1
    )

    rec_isbns = np.array(isbn_index)[indices[0][1:]]

    return books[books['ISBN'].isin(rec_isbns)][
        ['Book-Title', 'Book-Author', 'Image-URL-M']
    ]

# -------------------------------
# BUTTON ACTION
# -------------------------------
if st.button("Recommend"):
    results = recommend_svd_from_title(selected_book)

    if results is None or results.empty:
        st.warning("⚠ Not enough data to recommend similar books.")
    else:
        for _, row in results.iterrows():
            st.image(row['Image-URL-M'], width=120)
            st.write(f"**{row['Book-Title']}**")
            st.write(row['Book-Author'])
            st.markdown("---")
