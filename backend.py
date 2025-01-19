import pandas as pd
import numpy as np
import os
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Paths to dataset
credits_path = "archive/tmdb_5000_credits.csv"
movies_path = "archive/tmdb_5000_movies.csv"

# Ensure files exist
if not os.path.exists(credits_path):
    raise FileNotFoundError(f"File not found: {credits_path}")
if not os.path.exists(movies_path):
    raise FileNotFoundError(f"File not found: {movies_path}")

# Load datasets
credits = pd.read_csv(credits_path)
movies = pd.read_csv(movies_path)

# Merge datasets on title
merged_movies = movies.merge(credits, on='title')

# Select relevant columns
movies_df = merged_movies[['movie_id', 'title', 'overview', 'genres', 'cast', 'crew', 'keywords']]

# Drop NaN values and duplicates
movies_df = movies_df.dropna().drop_duplicates()

# Normalize movie titles to lowercase without leading/trailing spaces
movies_df['title'] = movies_df['title'].str.lower().str.strip()

# Helper function to parse JSON-like columns
def convert(obj):
    if isinstance(obj, str):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    return obj

# Apply to relevant columns
movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)

# Limit cast and crew to top 3 names
def convert3(obj):
    if isinstance(obj, str):
        obj = ast.literal_eval(obj)
    L = []
    for i in obj[:3]:  # Limit to top 3
        L.append(i['name'])
    return L

movies_df['cast'] = movies_df['cast'].apply(convert3)
movies_df['crew'] = movies_df['crew'].apply(convert3)

# Preprocessing: Remove spaces in names and split overview into words
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())

# Combine all features into a single 'tags' column
movies_df['tags'] = movies_df['cast'] + movies_df['crew'] + movies_df['genres'] + movies_df['keywords'] + movies_df['overview']

# Create new dataframe with processed data
new_df = movies_df[['movie_id', 'title', 'tags', 'genres']]

# Fix the setting copy issue by using .loc
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Text vectorization using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Stem words to reduce redundancy
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

# Apply stemming and fix using .loc
new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

# Calculate cosine similarity matrix
similarity = cosine_similarity(vectors)

# Ensure similarity matrix matches the number of movies
assert similarity.shape[0] == len(new_df), "Mismatch between similarity matrix and dataframe."
assert similarity.shape[1] == len(new_df), "Similarity matrix should be square."

# Save preprocessed data
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pickle.dump(new_df, open(os.path.join(output_dir, 'movies.pkl'), 'wb'))
pickle.dump(similarity, open(os.path.join(output_dir, 'similarity.pkl'), 'wb'))

# Recommendation function based on title
def recommend_by_title(title):
    try:
        # Normalize input title for case-insensitive matching
        title = title.lower().strip()
        
        # Match title in the dataframe
        matches = new_df[new_df['title'].str.contains(title, case=False)]
        if matches.empty:
            print(f"DEBUG: No matches found for title '{title}'")
            return [f"Movie '{title}' not found."]
        
        print(f"DEBUG: Matches found for title '{title}': {matches['title'].values}")
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
            print(f"DEBUG: No matches found for title '{title}'")
            return [f"Movie '{title}' not found."]
        
        movie_index = matches.index[0]
        movie_genres = set(new_df.iloc[movie_index].genres)
        
        # Find movies with similar genres
        similar_genre_movies = new_df[new_df['genres'].apply(lambda x: bool(set(x) & movie_genres))]
        similar_genre_movies = similar_genre_movies[similar_genre_movies.index != movie_index]
        
        return similar_genre_movies['title'].tolist()[:10]
    
    except Exception as e:
        return [f"Error in genre-based recommendation: {str(e)}"]

# Combined recommendation function
def recommend(title):
    print("Recommendations based on your input:")
    recommendations = recommend_by_title(title)
    print("\n".join(recommendations))

# Test the recommendation system
if __name__ == "__main__":
    test_title = input("Enter a movie title to get recommendations: ")
    recommendations = recommend_by_title(test_title)
    print("\nRecommendations for '{}':".format(test_title))
    print("\n".join(recommendations))
