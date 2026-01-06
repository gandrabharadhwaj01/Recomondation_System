import streamlit as st
import pickle
import numpy as np
from rapidfuzz import process

# LOADING THE  MODELS 
books = pickle.load(open("books.pkl", "rb"))
isbn_index = pickle.load(open("isbn_index.pkl", "rb"))
reduced_matrix = pickle.load(open("reduced_matrix.pkl", "rb"))
svd_knn = pickle.load(open("svd_knn.pkl", "rb"))

#  STREAMLIT UI 
st.title("📚 Book Recommendation System (SVD-Based CF)")

book_titles = books['Book-Title'].values
selected_book = st.selectbox("Select a Book", book_titles)

#  RECOMMEND FUNCTION 
def recommend_svd_from_title(book_title, top_n=5):

    # Fuzzy match
    match, score, _ = process.extractOne(book_title, book_titles)

    if score < 60:
        return None

    isbn = books.loc[
        books['Book-Title'] == match, 'ISBN'
    ].values[0]

    if isbn not in isbn_index:
        return None

    idx = list(isbn_index).index(isbn)

    distances, indices = svd_knn.kneighbors(
        reduced_matrix[idx].reshape(1, -1),
        n_neighbors=top_n + 1
    )

    rec_isbns = np.array(isbn_index)[indices[0][1:]]

    return books[books['ISBN'].isin(rec_isbns)][
        ['Book-Title', 'Book-Author', 'Image-URL-M']
    ]

#  BUTTON ACTION 
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
