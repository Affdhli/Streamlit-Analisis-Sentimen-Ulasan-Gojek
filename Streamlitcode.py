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

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Text preprocessing functions
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    return text

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    # Clean text
    cleaned = clean_text(text)
    # Tokenize
    tokens = word_tokenize(cleaned)
    # Remove stopwords
    filtered = remove_stopwords(tokens)
    # Stemming
    stemmed = stem_text(filtered)
    return ' '.join(stemmed)

# Load or initialize TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except:
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    # We'll fit it when we have data

# Load or initialize model
try:
    model = joblib.load('sentiment_model.pkl')
except:
    model = None
    st.warning("No trained model found. Please train a model first.")

# Streamlit app
st.title('Gojek Review Sentiment Analysis')

# Sidebar for navigation
menu = st.sidebar.selectbox('Menu', ['Home', 'Scrape Reviews', 'Preprocess Data', 'Train Model', 'Predict Sentiment'])

if menu == 'Home':
    st.header('Welcome to Gojek Review Analysis')
    st.write("""
    This app analyzes sentiment from Gojek app reviews on Google Play Store.
    
    Features:
    - Scrape reviews from Google Play Store
    - Preprocess text data
    - Train a sentiment analysis model
    - Predict sentiment of new reviews
    """)

elif menu == 'Scrape Reviews':
    st.header('Scrape Gojek Reviews')
    
    if st.button('Scrape Reviews from Google Play'):
        with st.spinner('Scraping reviews...'):
            result, _ = reviews(
                'com.gojek.app',
                lang='id',
                country='id',
                sort=Sort.MOST_RELEVANT,
                count=8000,  # Reduced for demo purposes
                filter_score_with=None
            )
            
            data = pd.DataFrame(np.array(result), columns=['review'])
            data = data.join(pd.DataFrame(data.pop('review').tolist()))
            
            # Keep only content and score columns
            data = data[['content', 'score']]
            
            # Save to session state
            st.session_state.reviews_data = data
            st.success(f"Successfully scraped {len(data)} reviews!")
        
        if 'reviews_data' in st.session_state:
            st.subheader('Sample Reviews')
            st.dataframe(st.session_state.reviews_data.head())

elif menu == 'Preprocess Data':
    st.header('Preprocess Review Data')
    
    if 'reviews_data' not in st.session_state:
        st.warning("Please scrape reviews first.")
    else:
        data = st.session_state.reviews_data.copy()
        
        st.subheader('Original Data')
        st.dataframe(data.head())
        
        if st.button('Preprocess Text'):
            with st.spinner('Preprocessing text...'):
                # Apply preprocessing
                data['cleaned_review'] = data['content'].apply(clean_text)
                
                # Save to session state
                st.session_state.processed_data = data
                st.success("Text preprocessing completed!")

        if 'processed_data' in st.session_state:
            st.subheader('Processed Data')
            st.dataframe(st.session_state.processed_data[['content', 'cleaned_text', 'score']].head())

elif menu == 'Train Model':
    st.header('Train Sentiment Model')
    
    if 'processed_data' not in st.session_state:
        st.warning("Please preprocess data first.")
    else:
        data = st.session_state.processed_data.copy()
        
        st.subheader('Training Data')
        st.write(f"Number of reviews: {len(data)}")
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        
        # Fit TF-IDF
        tfidf_vectorizer.fit(data['cleaned_text'])
        X = tfidf_vectorizer.transform(data['cleaned_text'])
        y = data['score']
        
        # Save vectorizer
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if st.button('Train Random Forest Model'):
            with st.spinner('Training model...'):
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                
                # Save model
                joblib.dump(model, 'sentiment_model.pkl')
                st.session_state.model = model
                st.success("Model training completed!")
                
                # Evaluate
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.subheader('Model Evaluation')
                st.dataframe(report_df)

elif menu == 'Predict Sentiment':
    st.header('Predict Sentiment of New Reviews')
    
    if model is None:
        st.warning("Please train a model first.")
    else:
        review_input = st.text_area("Enter a review to analyze:")
        
        if st.button('Analyze Sentiment'):
            if review_input:
                # Preprocess
                processed = preprocess_text(review_input)
                
                # Vectorize
                vectorized = tfidf_vectorizer.transform([processed])
                
                # Predict
                prediction = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]
                
                st.subheader('Prediction Result')
                st.write(f"Predicted Rating: {prediction} stars")
                
                # Show probabilities
                proba_df = pd.DataFrame({
                    'Rating': [1, 2, 3, 4, 5],
                    'Probability': proba
                })
                st.bar_chart(proba_df.set_index('Rating'))
            else:
                st.warning("Please enter a review to analyze.")
