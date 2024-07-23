import streamlit as st
import pickle
import pandas as pd
import re
import string
import base64
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix

# Function to load and apply the CSS file

page_bg_img="""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://plus.unsplash.com/premium_vector-1715632451165-87c3a13df4c1?q=80&w=580&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
        background-size: cover;
        
    }
    
    </style>
    """
st.markdown(page_bg_img, unsafe_allow_html=True)
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


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters







# Button to reset inputs

# with col2:
#     if st.button('Reset'):
#         st.new_review = ''
#         st.new_audience_score = 0
#         st.new_runtime_minutes = 0
#         st.new_genre = []
#         st.experimental_rerun()


