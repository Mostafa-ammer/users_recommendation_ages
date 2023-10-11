import streamlit as st
import surprise
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import StandardScaler
import surprise
from surprise import SVD
import tensorflow as tf
from surprise import accuracy
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from tensorflow.keras.models import Model

from wordcloud import WordCloud
from surprise import Dataset, Reader, KNNBasic
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model from the file
model = surprise.dump.load('knn_model.pkl')[1]
merged_data = pd.read_csv('merged_data.csv')
movies_data = pd.read_csv('movies.dat', names=["MovieID", "Title", "Genres"], sep="::")
users_data = pd.read_csv("users.dat", names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], sep="::")
group_criteria = {
    'Children': (merged_data['Age'] <= 12),
    'Teens': (merged_data['Age'] > 12) & (merged_data['Age'] <= 18),
    'Adults': (merged_data['Age'] > 18),
}

# Create a dictionary to map MovieID to Title
movie_id_to_title = dict(zip(movies_data['MovieID'], movies_data['Title']))



# Streamlit app
st.title("Movie Recommendations System")

# Input widgets
group_input = st.selectbox("Select a group:", list(group_criteria.keys()))
user_input = st.number_input("Enter a user ID:", min_value=1, max_value=users_data['UserID'].max())



# Function to get recommendations
def get_recommendations(group, user_id, top_n=10):
    if group not in group_criteria:
        return "Invalid group"

    user_in_group = merged_data[(merged_data['UserID'] == user_id) & group_criteria[group]].shape[0] > 0

    if not user_in_group:
        return "User not found in the specified group"

    user_history = merged_data[(merged_data['UserID'] == user_id) & (merged_data['Rating'] >= 4)]
    user_movie_ids = user_history['MovieID'].tolist()

    top_movie_ids = []

    if group == 'Children':
        genres_to_recommend = ['Children','Animation']
    elif group == 'Teens':
        genres_to_recommend = ['Drama', 'Action']
    elif group == 'Adults':
        genres_to_recommend = ['Romance','Comedy']
    else:
        genres_to_recommend = []

    for movie_id in range(1, 3953):
        if movie_id not in user_movie_ids:
            movie_genres = movies_data.loc[movies_data['MovieID'] == movie_id, 'Genres']
            if not movie_genres.empty and any(genre in movie_genres.values[0] for genre in genres_to_recommend):
                prediction = model.predict(user_id, movie_id)
                top_movie_ids.append((movie_id, prediction.est))

    top_movie_ids.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie[0] for movie in top_movie_ids[:top_n]]

    top_movie_titles = [movie_id_to_title[movie_id].rsplit('(', 1)[0].strip() for movie_id in top_movie_ids]

    return top_movie_titles

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(group_input, user_input)
    if isinstance(recommendations, list):
        st.write(f"Top {len(recommendations)} Movie Recommendations for {group_input} Group and User {user_input}:")
        for movie_title in recommendations:
            st.write(movie_title)
    else:
        st.write(recommendations)
