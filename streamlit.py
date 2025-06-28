import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
from google_play_scraper import Sort, reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Analisis Sentimen Gojek", layout="wide")

# Sidebar
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Scraping Data", "Preprocessing", "Training Model", "Prediksi Teks"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Fungsi-fungsi preprocessing
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

def stem_tokens(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in tokens]

# Halaman Scraping Data
if app_mode == "Scraping Data":
    st.title("Scraping Data Ulasan Gojek dari Play Store")
    
    st.write("""
    Aplikasi ini akan melakukan scraping ulasan Gojek dari Google Play Store.
    Masukkan parameter di bawah ini untuk memulai proses scraping.
    """)
    
    count = st.number_input("Jumlah Ulasan yang Akan Diambil", min_value=100, max_value=8000, value=1000)
    
    if st.button("Mulai Scraping"):
        with st.spinner('Sedang melakukan scraping...'):
            try:
                result, _ = reviews(
                    'com.gojek.app',
                    lang='id',
                    country='id',
                    sort=Sort.MOST_RELEVANT,
                    count=count,
                    filter_score_with=None
                )
                
                data = pd.DataFrame(np.array(result), columns=['review'])
                data = data.join(pd.DataFrame(data.pop('review').tolist()))
                data = data[['content', 'score']]
                
                # Labeling (tanpa netral)
                def label_sentiment(score):
                    return 'positif' if score >= 4 else 'negatif'
                
                data['label'] = data['score'].apply(label_sentiment)
                
                st.session_state.data = data
                
                st.success(f"Berhasil mengambil {len(data)} ulasan (hanya positif dan negatif)!")
                st.dataframe(data.head())
                
                # Visualisasi distribusi label
                fig, ax = plt.subplots()
                data['label'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
                ax.set_title('Distribusi Sentimen (Biner)')
                ax.set_xlabel('Sentimen')
                ax.set_ylabel('Jumlah Ulasan')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Terjadi error: {str(e)}")

# Halaman Preprocessing
elif app_mode == "Preprocessing":
    st.title("Preprocessing Data")
    
    if st.session_state.data is None:
        st.warning("Silakan lakukan scraping data terlebih dahulu di menu Scraping Data")
    else:
        data = st.session_state.data.copy()
        
        st.write("### Data Mentah")
        st.write(f"Jumlah data: {len(data)} (Positif: {len(data[data['label']=='positif'])}, Negatif: {len(data[data['label']=='negatif'])})")
        st.dataframe(data.head())
        
        if st.button("Mulai Preprocessing"):
            with st.spinner('Sedang memproses data...'):
                # Cleaning
                data['cleaned_text'] = data['content'].apply(clean_text)
                
                # Case folding
                data['case_folded'] = data['cleaned_text'].str.lower()
                
                # Tokenisasi
                data['tokens'] = data['case_folded'].apply(word_tokenize)
                
                # Stopword removal
                data['filtered_tokens'] = data['tokens'].apply(remove_stopwords)
                
                # Stemming
                data['stemmed_tokens'] = data['filtered_tokens'].apply(stem_tokens)
                
                # Gabungkan tokens
                data['processed_text'] = data['stemmed_tokens'].apply(lambda x: ' '.join(x))
                
                st.session_state.data = data
                
                st.success("Preprocessing selesai!")
                
                st.write("### Hasil Preprocessing")
                st.dataframe(data[['content', 'processed_text', 'label']].head())
                
                # Tampilkan contoh preprocessing
                st.write("### Contoh Tahapan Preprocessing")
                sample_idx = st.slider("Pilih contoh data", 0, len(data)-1, 0)
                sample = data.iloc[sample_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Text:**")
                    st.write(sample['content'])
                    
                    st.write("**Cleaned Text:**")
                    st.write(sample['cleaned_text'])
                    
                    st.write("**Case Folded:**")
                    st.write(sample['case_folded'])
                
                with col2:
                    st.write("**Tokens:**")
                    st.write(sample['tokens'])
                    
                    st.write("**Filtered Tokens (no stopwords):**")
                    st.write(sample['filtered_tokens'])
                    
                    st.write("**Stemmed Tokens:**")
                    st.write(sample['stemmed_tokens'])

# Halaman Training Model
elif app_mode == "Training Model":
    st.title("Training Model SVM (Biner)")
    
    if st.session_state.data is None:
        st.warning("Silakan lakukan scraping dan preprocessing data terlebih dahulu")
    else:
        data = st.session_state.data.copy()
        
        st.write("### Distribusi Label")
        st.write(data['label'].value_counts())
        
        st.write("### Parameter Model")
        max_features = st.number_input("Jumlah Maksimal Fitur TF-IDF", min_value=100, max_value=5000, value=3000)
        test_size = st.slider("Persentase Data Testing", 10, 40, 20) / 100
        
        if st.button("Mulai Training"):
            with st.spinner('Sedang melatih model...'):
                try:
                    # Ekstraksi fitur TF-IDF
                    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
                    tfidf_features = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
                    
                    # Simpan vectorizer
                    st.session_state.vectorizer = tfidf_vectorizer
                    
                    # Bagi data
                    X = tfidf_features
                    y = data['label']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    # Training SVM
                    svm_model = SVC(kernel='linear', probability=True, random_state=42)
                    svm_model.fit(X_train, y_train)
                    
                    # Simpan model
                    st.session_state.model = svm_model
                    
                    st.success("Model berhasil dilatih!")
                    
                    # Evaluasi
                    st.write("### Evaluasi Model")
                    
                    # Akurasi training
                    y_train_pred = svm_model.predict(X_train)
                    train_acc = accuracy_score(y_train, y_train_pred)
                    
                    # Akurasi testing
                    y_test_pred = svm_model.predict(X_test)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Akurasi Training", f"{train_acc:.2%}")
                    
                    with col2:
                        st.metric("Akurasi Testing", f"{test_acc:.2%}")
                    
                    # Classification report
                    st.write("#### Classification Report")
                    report = classification_report(y_test, y_test_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    
                    # Confusion matrix
                    st.write("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_test_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                              xticklabels=['negatif', 'positif'], 
                              yticklabels=['negatif', 'positif'], ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix (Biner)')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Terjadi error: {str(e)}")

# Halaman Prediksi Teks
elif app_mode == "Prediksi Teks":
    st.title("Prediksi Sentimen Teks (Biner)")
    
    if st.session_state.model is None or st.session_state.vectorizer is None:
        st.warning("Silakan latih model terlebih dahulu di menu Training Model")
    else:
        model = st.session_state.model
        vectorizer = st.session_state.vectorizer
        
        st.write("""
        Masukkan teks ulasan di bawah ini untuk memprediksi sentimennya.
        Aplikasi akan mengklasifikasikan apakah ulasan tersebut positif atau negatif.
        """)
        
        input_text = st.text_area("Masukkan Teks Ulasan", "Aplikasi Gojek sangat bagus dan bermanfaat")
        
        if st.button("Prediksi Sentimen"):
            with st.spinner('Menganalisis sentimen...'):
                try:
                    # Preprocessing
                    cleaned_text = clean_text(input_text)
                    case_folded = cleaned_text.lower()
                    tokens = word_tokenize(case_folded)
                    filtered_tokens = remove_stopwords(tokens)
                    stemmed_tokens = stem_tokens(filtered_tokens)
                    processed_text = ' '.join(stemmed_tokens)
                    
                    # Ekstraksi fitur
                    tfidf_features = vectorizer.transform([processed_text]).toarray()
                    
                    # Prediksi
                    prediction = model.predict(tfidf_features)[0]
                    probability = model.predict_proba(tfidf_features)[0]
                    
                    # Tampilkan hasil
                    st.write("### Hasil Prediksi")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 'positif':
                            st.success(f"Sentimen: {prediction.capitalize()}")
                        else:
                            st.error(f"Sentimen: {prediction.capitalize()}")
                    
                    with col2:
                        st.metric("Probabilitas", 
                                 f"Positif: {probability[1]:.2%}\nNegatif: {probability[0]:.2%}")
                    
                    # Visualisasi probabilitas
                    fig, ax = plt.subplots()
                    ax.bar(['Negatif', 'Positif'], probability, color=['red', 'green'])
                    ax.set_title('Probabilitas Sentimen')
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Terjadi error: {str(e)}")

# Informasi tambahan
st.sidebar.markdown("---")
st.sidebar.info("""
Aplikasi Analisis Sentimen Ulasan Gojek (Biner)\n
Hanya klasifikasi Positif/Negatif\n
Menggunakan:
- Scraping data dari Google Play Store
- Preprocessing text
- Ekstraksi fitur TF-IDF
- Klasifikasi dengan SVM
""")
