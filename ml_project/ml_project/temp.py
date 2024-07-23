import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.base import BaseEstimator, TransformerMixin

movies_list=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_list)
movies_review=pickle.load(open('movie_review.pkl','rb'))
review=pd.DataFrame(movies_review)
review_sentiment=pickle.load(open('svc_model.pkl','rb'))

st.title('Classification of Movie Reviews')

# Create a mapping from original movie IDs to modified ones
original_movie_ids = review['movieid'].unique()
modified_movie_ids = [movie_id.replace('_', ' ').upper() for movie_id in original_movie_ids]
movie_id_map = dict(zip(modified_movie_ids, original_movie_ids))

option=st.selectbox(
    'Type or select a movie from the dropdown',
    modified_movie_ids
)

# Get the original movie ID from the selected modified movie ID
selected_movie_id = movie_id_map[option]


# Search button to display reviews and sentiments
if st.button('Search'):
    st.write(f'Searching reviews for: {option}')
    
    # Filter reviews for the selected movie

    movie_rws = review[review['movieid'] == selected_movie_id]
    
    # Debug print to ensure the correct filtering
    st.write(f'Found {len(movie_rws)} reviews for {option}')

    # Display reviews and their sentiments
    st.write('### Reviews and Sentiments')
    if not movie_rws.empty:
        movie_rws['sentiment'] = review['sentiment']
        movie_rws['reviewText'] = review['reviewText']
        st.table(movie_rws[['reviewText', 'sentiment']])
    else:
        st.write('No reviews found for this movie.')


# Define the stemming function
def stemming(text):
    text = str(text)
    stemmed_text = re.sub("[^a-zA-Z]", " ", text)
    stemmed_text = stemmed_text.lower()
    stemmed_text = stemmed_text.split()
    stemmed_text = [word for word in stemmed_text if ((word == "not") or (word not in ENGLISH_STOP_WORDS))]
    stemmed_text = [word for word in stemmed_text if len(word) >= 3]
    stemmed_text = " ".join(stemmed_text)
    return stemmed_text

# Define the custom transformer
class CustomStemmer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.apply(stemming)
        return X_copy
    
# Create an instance of the custom transformer
stemmer = CustomStemmer()

# Define the TfidfVectorizer
reviewText_vectorizer = TfidfVectorizer()

# Create a pipeline for preprocessing
preprocessing_pipeline = Pipeline([
    ('stemmer', stemmer),
    ('vectorizer', reviewText_vectorizer),
    ('tfidf', TfidfVectorizer())
])
preprocessing_pipeline.fit(review['reviewText'])
# Function to predict sentiment
def predict_sentiment(review_text):
    if 'review_sentiment' not in globals():
        st.error('Sentiment analysis model is not loaded.')
        return None
    # Preprocess the review text

    preprocessed_text = preprocessing_pipeline.transform(pd.Series([review_text]))

    # Predict sentiment
    return review_sentiment.predict(preprocessed_text)[0]

# Input field for new review and predict button
st.write('### Add a New Review')
new_review = st.text_area('Enter your review here:')
if st.button('Predict Sentiment'):
    if new_review:
        new_sentiment = predict_sentiment(new_review)
        
        if new_sentiment is not None:
            # Update the review DataFrame with the new review
            new_review_data = pd.DataFrame({'movieid': [selected_movie_id], 'reviewText': [new_review], 'sentiment': [new_sentiment]})
            review = pd.concat([review, new_review_data], ignore_index=True)
            
            st.write('### New Review and Sentiment')
            st.write(pd.DataFrame({'review': [new_review], 'sentiment': [new_sentiment]}))
            
            # Display updated reviews and sentiments
            st.write('### Updated Reviews and Sentiments')
            movie_rws = review[review['movieid'] == selected_movie_id]
            st.table(movie_rws[['reviewText', 'sentiment']])
    else:
        st.write('Please enter a review to predict the sentiment.')
