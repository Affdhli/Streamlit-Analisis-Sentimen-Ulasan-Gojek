import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and layout
st.set_page_config(page_title="Gojek Sentiment Analysis", layout="wide")
st.title("PENERAPAN SVM PADA ANALISIS SENTIMEN ULASAN PENGGUNA GOJEK DI PLAY STORE DENGAN PENDEKATAN TF-IDF")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Text preprocessing functions
def clean_text(text):
    text = re.sub(r'[^\\w\\s]', '', text)
    text = text.lower()
    text = re.sub(r'\\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text, re.I|re.A)
    return text

def case_fold(text):
    return text.lower()

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    # Clean text
    cleaned = clean_text(text)
    # Case folding
    case_folded = case_fold(cleaned)
    # Tokenization
    tokens = word_tokenize(case_folded)
    # Stopword removal
    filtered = remove_stopwords(tokens)
    # Stemming
    stemmed = stem_text(filtered)
    return ' '.join(stemmed)

# Sidebar for navigation
st.sidebar.title("Navigasi")
options = st.sidebar.radio("Pilih Halaman:", 
                          ["Data Awal", "Preprocessing", "TF-IDF", "Model Training & Evaluation", "Prediksi"])

# Load data (you would replace this with your actual data loading)
@st.cache_data
def load_data():
    # In a real app, you would load your scraped data here
    # For demo purposes, we'll create a sample DataFrame
    data = pd.DataFrame({
        'content': [
            "terlalu terlalu terlalu... apk yg tidak bisa dipercaya kuota cepat abis update terus",
            "Aplikasinya bagus kok kalo malam pengen makan rumah makan buka",
            "saya kecewa dg layanan gojek, padahal saya mau pesan tiba2 dibatalkan",
            "Pengalaman buruk dan kecewa ketika pesan Go Car pulang kerja",
            "saya suka aplikasi ini murah cepat dan memudahkan perjalanan"
        ],
        'score': [1, 5, 1, 2, 5]
    })
    return data

data = load_data()

if options == "Data Awal":
    st.header("Data Ulasan Awal")
    st.write("Berikut adalah contoh data ulasan pengguna Gojek dari Play Store:")
    st.dataframe(data)
    
    st.subheader("Distribusi Rating")
    fig, ax = plt.subplots()
    data['score'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah Ulasan")
    ax.set_title("Distribusi Rating Ulasan")
    st.pyplot(fig)

elif options == "Preprocessing":
    st.header("Hasil Preprocessing Teks")
    st.write("Berikut adalah tahapan preprocessing yang dilakukan pada teks ulasan:")
    
    # Add preprocessing steps to the DataFrame
    data['cleaned'] = data['content'].apply(clean_text)
    data['case_folded'] = data['cleaned'].apply(case_fold)
    data['tokens'] = data['cleaned'].apply(word_tokenize)
    data['filtered_tokens'] = data['tokens'].apply(remove_stopwords)
    data['stemmed_tokens'] = data['filtered_tokens'].apply(stem_text)
    data['processed_text'] = data['stemmed_tokens'].apply(lambda x: ' '.join(x))
    
    st.subheader("Contoh Hasil Setiap Tahap Preprocessing")
    st.dataframe(data[['content', 'cleaned', 'case_folded', 'tokens', 'filtered_tokens', 'stemmed_tokens', 'processed_text']].head())
    
    st.subheader("Visualisasi Preprocessing")
    sample_text = st.selectbox("Pilih contoh teks untuk melihat proses preprocessing:", data['content'].tolist())
    
    if sample_text:
        st.write("**Teks Asli:**")
        st.write(sample_text)
        
        cleaned = clean_text(sample_text)
        st.write("**Setelah Cleaning:**")
        st.write(cleaned)
        
        case_folded = case_fold(cleaned)
        st.write("**Setelah Case Folding:**")
        st.write(case_folded)
        
        tokens = word_tokenize(case_folded)
        st.write("**Setelah Tokenization:**")
        st.write(tokens)
        
        filtered = remove_stopwords(tokens)
        st.write("**Setelah Stopword Removal:**")
        st.write(filtered)
        
        stemmed = stem_text(filtered)
        st.write("**Setelah Stemming:**")
        st.write(stemmed)
        
        processed = ' '.join(stemmed)
        st.write("**Teks Hasil Preprocessing:**")
        st.write(processed)

elif options == "TF-IDF":
    st.header("Ekstraksi Fitur dengan TF-IDF")
    st.write("Pada tahap ini, teks yang sudah diproses akan diubah menjadi vektor fitur menggunakan TF-IDF.")
    
    # Check if preprocessing has been done
    if 'processed_text' not in data.columns:
        st.warning("Silakan lakukan preprocessing terlebih dahulu di halaman Preprocessing.")
    else:
        # Initialize TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=300)
        tfidf_features = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_vectorizer.get_feature_names_out())
        
        st.subheader("Matriks TF-IDF")
        st.write("Berikut adalah contoh beberapa fitur TF-IDF:")
        st.dataframe(tfidf_df.head())
        
        st.subheader("Fitur TF-IDF Paling Penting")
        # Get top features
        sum_tfidf = tfidf_features.sum(axis=0)
        feature_importance = pd.DataFrame([(word, sum_tfidf[idx]) for word, idx in tfidf_vectorizer.vocabulary_.items()], 
                                        columns=['feature', 'importance']).sort_values('importance', ascending=False)
        
        # Plot top features
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20), ax=ax)
        ax.set_title("20 Fitur TF-IDF dengan Nilai Tertinggi")
        st.pyplot(fig)
        
        # Save TF-IDF vectorizer and features for later use
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        joblib.dump(tfidf_df, 'tfidf_features.pkl')

elif options == "Model Training & Evaluation":
    st.header("Pelatihan dan Evaluasi Model SVM")
    st.write("Pada tahap ini, kita akan melatih model SVM dan mengevaluasi performanya.")
    
    # Check if TF-IDF features are available
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        tfidf_features = joblib.load('tfidf_features.pkl')
    except:
        st.warning("Silakan ekstrak fitur TF-IDF terlebih dahulu di halaman TF-IDF.")
        st.stop()
    
    # Prepare data
    X = tfidf_features
    y = data['score'].apply(lambda x: 1 if x > 3 else 0)  # Convert to binary sentiment (1=positive, 0=negative)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Display results
    st.subheader("Hasil Evaluasi Model")
    st.write(f"**Akurasi Model:** {accuracy:.2f}")
    
    st.write("**Classification Report:**")
    st.text(report)
    
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    # Save model
    joblib.dump(svm_model, 'svm_model.pkl')
    st.success("Model berhasil disimpan!")

elif options == "Prediksi":
    st.header("Prediksi Sentimen Ulasan Baru")
    st.write("Gunakan model yang telah dilatih untuk memprediksi sentimen ulasan baru.")
    
    # Load model and vectorizer
    try:
        svm_model = joblib.load('svm_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except:
        st.warning("Silakan latih model terlebih dahulu di halaman Model Training & Evaluation.")
        st.stop()
    
    # Input text
    user_input = st.text_area("Masukkan ulasan yang ingin dianalisis:", "Aplikasi Gojek sangat membantu!")
    
    if st.button("Prediksi Sentimen"):
        # Preprocess the input
        processed_input = preprocess_text(user_input)
        
        # Transform using TF-IDF
        input_tfidf = tfidf_vectorizer.transform([processed_input])
        
        # Make prediction
        prediction = svm_model.predict(input_tfidf)
        
        # Display result
        st.subheader("Hasil Prediksi")
        st.write(f"**Ulasan:** {user_input}")
        st.write(f"**Sentimen Prediksi:** {'Positif' if prediction[0] == 1 else 'Negatif'}")
        
        # Show preprocessing steps
        with st.expander("Lihat Detail Preprocessing"):
            st.write("**Teks Asli:**")
            st.write(user_input)
            
            cleaned = clean_text(user_input)
            st.write("**Setelah Cleaning:**")
            st.write(cleaned)
            
            case_folded = case_fold(cleaned)
            st.write("**Setelah Case Folding:**")
            st.write(case_folded)
            
            tokens = word_tokenize(case_folded)
            st.write("**Setelah Tokenization:**")
            st.write(tokens)
            
            filtered = remove_stopwords(tokens)
            st.write("**Setelah Stopword Removal:**")
            st.write(filtered)
            
            stemmed = stem_text(filtered)
            st.write("**Setelah Stemming:**")
            st.write(stemmed)
            
            processed = ' '.join(stemmed)
            st.write("**Teks Hasil Preprocessing:**")
            st.write(processed)
