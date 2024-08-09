import streamlit as st
import pickle
import pandas as pd
import re
import string
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix


#load the model
svc_model = pickle.load(open(r'C:\new_ml_project\ml_project\ml_project\svc_model.pkl', 'rb'))

movies_list = pickle.load(open(r'C:\new_ml_project\ml_project\ml_project\movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_list)

tfidf_vectorizer = pickle.load(open(r'C:\new_ml_project\ml_project\ml_project\vectorizer.pkl', 'rb'))
label=pickle.load(open(r'C:\new_ml_project\ml_project\ml_project\multi_label.pkl','rb'))

movies_review = pickle.load(open(r'C:\new_ml_project\ml_project\ml_project\movie_review.pkl', 'rb'))
review = pd.DataFrame(movies_review)

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.findall(r'\b\w+\b', text)
        text = ['not_' + word if text[i - 1] == 'not' else word for i, word in enumerate(text)]
        text = [word for word in text if word not in ENGLISH_STOP_WORDS]
        return ' '.join(text)
    return ''


# # Streamlit app
st.title('Classification of Movie Reviews') 

# Movie selection
original_movie_ids = review['movieid'].unique()
modified_movie_ids = [movie_id.replace('_', ' ').upper() for movie_id in original_movie_ids]
movie_id_map = dict(zip(modified_movie_ids, original_movie_ids))

option = st.selectbox('Type or select a movie from the dropdown', modified_movie_ids)
selected_movie_id = movie_id_map[option]

# Search button to display reviews and sentiments
if st.button('Search'):
    st.write(f'Searching reviews for: {option}')
    movie_rws = review[review['movieid'] == selected_movie_id]
    st.write(f'Found {len(movie_rws)} reviews for {option}')
    if not movie_rws.empty:
        movie_rws['sentiment'] = review['sentiment']
        movie_rws['reviewText'] = review['reviewText']
        st.table(movie_rws[['reviewText', 'sentiment']])
    else:
        st.write('No reviews found for this movie.')
# Input fields for new review
st.write('### Add a New Review')
new_review = st.text_area('Enter your review here:')
preprocess_review=preprocess_text(new_review)
# print("PREPROCESS_TEXT",preprocess_review)
reviewtext_data=tfidf_vectorizer.transform([preprocess_review])
# print("REVIEW-TEXT-DATA",reviewtext_data)

new_audience_score = st.number_input('Enter Audience Score (0-100):', min_value=0, max_value=100)
new_runtime_minutes = st.number_input('Enter Runtime in Minutes:', min_value=0)

audienceScore_scaler=MinMaxScaler()
runtimeMinutes_scaler=MinMaxScaler()
X_df = pd.DataFrame({
    'audienceScore': [85, 90, 75, 60],
    'runtimeMinutes': [120, 150, 90, 110]
})

# Fit the scalers on the original DataFrame
audienceScore_scaler.fit(X_df[['audienceScore']])
runtimeMinutes_scaler.fit(X_df[['runtimeMinutes']])

# Transform the new values using the fitted scalers
scaled_audience_score = audienceScore_scaler.transform([[new_audience_score]])
scaled_runtime_minutes = runtimeMinutes_scaler.transform([[new_runtime_minutes]])
#Combine the transformed values into a DataFrame
numerical_data = pd.DataFrame({
    'audienceScore': scaled_audience_score.flatten(),
    'runtimeMinutes': scaled_runtime_minutes.flatten()
})

# print("NUMERICAL-DATA",numerical_data)

new_genre = st.multiselect('Select Genre(s):', options=movies["genre"].unique())
genre_data=label.transform([new_genre])
# print("GENRE-DATA",genre_data)


def predict_sentiment(reviewtext_data, genre_data,numerical_data):
    reviewtext_data_dense = reviewtext_data.toarray()
    X_df_matrix = hstack([csr_matrix(reviewtext_data_dense), genre_data, numerical_data])
    return svc_model.predict(X_df_matrix)


col1, col2 = st.columns([4, 1])
with col1:
    if st.button('Predict Sentiment'):
        if new_review and new_genre:
            sentiment = predict_sentiment(reviewtext_data, genre_data,numerical_data)
            st.write('### New Review and Sentiment')
            st.write(pd.DataFrame({'review': [new_review], 'sentiment': [sentiment]}))
        else:
            st.write('Please enter a review and select genre(s) to predict the sentiment.')

# Button to reset inputs

# with col2:
#     if st.button('Reset'):
#         st.new_review = ''
#         st.new_audience_score = 0
#         st.new_runtime_minutes = 0
#         st.new_genre = []
#         st.experimental_rerun()


