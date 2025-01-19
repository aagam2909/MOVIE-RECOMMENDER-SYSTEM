import pandas as pd
import numpy as np
import os
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import streamlit as st

# Paths to dataset and preprocessed files
output_dir = 'data'
movies_file = os.path.join(output_dir, 'movies.pkl')
similarity_file = os.path.join(output_dir, 'similarity.pkl')

# Load preprocessed data if available
if not os.path.exists(movies_file) or not os.path.exists(similarity_file):
    raise FileNotFoundError("Preprocessed files not found. Please preprocess the data first.")

# Load preprocessed files
new_df = pickle.load(open(movies_file, 'rb'))
similarity = pickle.load(open(similarity_file, 'rb'))

# Recommendation function based on title
def recommend_by_title(title):
    try:
        # Normalize input title for case-insensitive matching
        title = title.lower().strip()
        
        # Match title in the dataframe
        matches = new_df[new_df['title'].str.contains(title, case=False)]
        if matches.empty:
            return [f"Movie '{title}' not found."]
        
        movie_index = matches.index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
        
        return [new_df.iloc[i[0]].title for i in movies_list]
    
    except Exception as e:
        return [f"Error in recommendation: {str(e)}"]

# Recommendation function based on genre
def recommend_by_genre(title):
    try:
        # Normalize input title for case-insensitive matching
        title = title.lower().strip()
        
        # Match title in the dataframe
        matches = new_df[new_df['title'].str.contains(title, case=False)]
        if matches.empty:
            return [f"Movie '{title}' not found."]
        
        movie_index = matches.index[0]
        movie_genres = set(new_df.iloc[movie_index].genres)
        
        # Find movies with similar genres
        similar_genre_movies = new_df[new_df['genres'].apply(lambda x: bool(set(x) & movie_genres))]
        similar_genre_movies = similar_genre_movies[similar_genre_movies.index != movie_index]
        
        return similar_genre_movies['title'].tolist()[:10]
    
    except Exception as e:
        return [f"Error in genre-based recommendation: {str(e)}"]

# Streamlit app interface
st.title("Movie Recommendation System")

# Input section for movie title
title_input = st.text_input("Enter a movie title to get recommendations:")

if title_input:
    # Recommendation based on title
    st.subheader(f"Recommendations based on title: '{title_input}'")
    recommendations = recommend_by_title(title_input)
    for movie in recommendations:
        st.write(movie)

    # Recommendation based on genre
    st.subheader(f"Genre-based recommendations for '{title_input}'")
    genre_recommendations = recommend_by_genre(title_input)
    for movie in genre_recommendations:
        st.write(movie)

else:
    st.write("Please enter a movie title to see recommendations.")
