import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from google_play_scraper import Sort, reviews
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Gojek Review Analysis", layout="wide")

# Title and description
st.title("Gojek App Review Analysis")
st.write("This app analyzes sentiment from Gojek app reviews on Google Play Store")

# Sidebar for user inputs
with st.sidebar:
    st.header("Configuration")
    num_reviews = st.slider("Number of reviews to fetch", 100, 8000, 1000, 100)
    analyze_button = st.button("Analyze Reviews")

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Text processing functions
def clean_text(text):
    text = re.sub(r'[^\\w\\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z\\s]', '', text, re.I|re.A)
    return text

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    return [stemmer.stem(word) for word in tokens]

# Main function to process reviews
def analyze_reviews(num_reviews):
    # Scrape reviews
    st.write("Fetching reviews from Google Play Store...")
    result, _ = reviews(
        'com.gojek.app',
        lang='id',
        country='id',
        sort=Sort.MOST_RELEVANT,
        count=num_reviews,
        filter_score_with=None
    )
    
    # Create DataFrame
    data = pd.DataFrame(np.array(result), columns=['review'])
    data = data.join(pd.DataFrame(data.pop('review').tolist()))
    
    # Keep only necessary columns
    data = data[['content', 'score']]
    
    # Text preprocessing
    st.write("Processing reviews...")
    data['cleaned_review'] = data['content'].apply(clean_text)
    data['tokens'] = data['cleaned_review'].apply(word_tokenize)
    data['filtered_tokens'] = data['tokens'].apply(remove_stopwords)
    data['stemmed_tokens'] = data['filtered_tokens'].apply(stem_text)
    data['processed_text'] = data['stemmed_tokens'].apply(lambda tokens: ' '.join(tokens))
    
    return data

# Display results
def display_results(data):
    st.subheader("Processed Reviews")
    st.dataframe(data.head(10))
    
    # Sentiment distribution
    st.subheader("Review Score Distribution")
    score_counts = data['score'].value_counts().sort_index()
    st.bar_chart(score_counts)
    
    # TF-IDF Vectorization
    st.subheader("TF-IDF Features")
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    tfidf_features = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
    tfidf_data = pd.DataFrame(tfidf_features, columns=tfidf_vectorizer.get_feature_names_out())
    st.dataframe(tfidf_data.head(10))
    
    # Save processed data
    if st.button("Save Processed Data"):
        data.to_csv('processed_reviews.csv', index=False)
        st.success("Data saved successfully!")

# Main app logic
if analyze_button:
    with st.spinner("Analyzing reviews..."):
        review_data = analyze_reviews(num_reviews)
        display_results(review_data)
else:
    st.info("Configure the settings in the sidebar and click 'Analyze Reviews' to get started.")

# Add some explanations
st.markdown("""
### About This App
This application performs sentiment analysis on Gojek app reviews from Google Play Store by:
1. Scraping reviews using google-play-scraper
2. Cleaning and preprocessing the text (removing punctuation, stopwords, etc.)
3. Performing stemming using Sastrawi
4. Extracting features using TF-IDF
5. Displaying the results

The review scores (1-5 stars) are used as the sentiment indicators.
""")
