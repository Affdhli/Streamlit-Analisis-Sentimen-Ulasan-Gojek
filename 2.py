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

# Fungsi untuk mengatur tampilan Streamlit
def set_page_config():
    st.set_page_config(
        page_title="Analisis Sentimen Ulasan Gojek",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

# Fungsi untuk preprocessing text
def preprocess_text(text):
    # Clean text
    text = re.sub(r'[^\w\s]', '', str(text))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return ' '.join(stemmed_tokens)

# Fungsi utama
def main():
    set_page_config()
    st.title("🚗 Analisis Sentimen Ulasan Gojek")
    st.write("Aplikasi ini menganalisis sentimen ulasan Gojek dari Play Store")
    
    # Inisialisasi session state
    if 'reviews_data' not in st.session_state:
        st.session_state.reviews_data = None
    
    # Sidebar
    st.sidebar.header("Pengaturan")
    sample_size = st.sidebar.slider("Jumlah data", 1000, 10000, 8000, 1000)
    
    tab1, tab2, tab3 = st.tabs(["Scraping Data", "Analisis Sentimen", "Klasifikasi Teks"])
    
    with tab1:
        st.header("📥 Scraping Data dari Play Store")
        
        if st.button('Scrape Reviews from Google Play'):
            with st.spinner('Scraping reviews...'):
                try:
                    result, _ = reviews(
                        'com.gojek.app',
                        lang='id',
                        country='id',
                        sort=Sort.MOST_RELEVANT,
                        count=8000,
                        filter_score_with=None
                    )
                    
                    # Konversi hasil scraping ke DataFrame
                    data = pd.DataFrame(np.array(result), columns=['review'])
                    data = data.join(pd.DataFrame(data.pop('review').tolist()))
                    
                    # Menyimpan hanya kolom yang diperlukan
                    data = data[['content', 'score']]
                    
                    # Simpan ke session state
                    st.session_state.reviews_data = data
                    st.success(f"Berhasil mengambil {len(data)} ulasan!")
                    
                    # Simpan ke file CSV
                    data.to_csv('gojek_reviews.csv', index=False)
                    st.info("Data telah disimpan ke file 'gojek_reviews.csv'")
                    
                except Exception as e:
                    st.error(f"Gagal mengambil data: {str(e)}")
        
        if st.button("Muat Data yang Tersimpan"):
            try:
                data = pd.read_csv('gojek_reviews.csv')
                st.session_state.reviews_data = data
                st.success(f"Berhasil memuat {len(data)} ulasan!")
            except Exception as e:
                st.warning(f"File data tidak ditemukan atau error: {str(e)}")
        
        if st.session_state.reviews_data is not None:
            st.subheader("Preview Data")
            st.dataframe(st.session_state.reviews_data.head(10))
            
            # Visualisasi distribusi rating
            st.subheader("Distribusi Rating")
            fig, ax = plt.subplots()
            sns.countplot(x='score', data=st.session_state.reviews_data, ax=ax)
            st.pyplot(fig)
    
    with tab2:
        st.header("📊 Analisis Sentimen")
        
        if st.session_state.reviews_data is None:
            st.warning("Silakan muat atau ambil data terlebih dahulu di tab Scraping Data")
            return
        
        with st.spinner("Memproses data..."):
            data = st.session_state.reviews_data.copy()
            
            # Labeling
            data['label'] = data['score'].apply(lambda x: 'positif' if x >= 4 else ('netral' if x == 3 else 'negatif'))
            data = data[data['label'] != 'netral']
            
            # Preprocessing
            data['processed_text'] = data['content'].apply(preprocess_text)
            
            # TF-IDF
            tfidf = TfidfVectorizer(max_features=3000)
            X = tfidf.fit_transform(data['processed_text'])
            y = data['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = SVC(kernel='linear', probability=True, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            joblib.dump(model, 'svm_model.pkl')
            joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
        
        st.success("Proses analisis selesai!")
        st.metric("Akurasi Model", f"{accuracy:.2%}")
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Negatif', 'Positif'], 
                    yticklabels=['Negatif', 'Positif'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with tab3:
        st.header("🔎 Klasifikasi Teks")
        
        try:
            model = joblib.load('svm_model.pkl')
            tfidf = joblib.load('tfidf_vectorizer.pkl')
            st.success("Model berhasil dimuat!")
        except:
            st.error("Model belum tersedia. Silakan lakukan analisis di tab Analisis Sentimen terlebih dahulu.")
            return
        
        text_input = st.text_area("Masukkan teks untuk diklasifikasikan:", "Aplikasi Gojek sangat membantu!")
        
        if st.button("Klasifikasikan"):
            with st.spinner("Memproses..."):
                # Preprocess
                processed_text = preprocess_text(text_input)
                
                # Transform
                features = tfidf.transform([processed_text])
                
                # Predict
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                # Display results
                st.subheader("Hasil Klasifikasi")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sentimen", prediction.capitalize())
                
                with col2:
                    st.write("Probabilitas:")
                    prob_df = pd.DataFrame({
                        'Sentimen': ['Negatif', 'Positif'],
                        'Probabilitas': probabilities
                    })
                    st.bar_chart(prob_df.set_index('Sentimen'))
                
                st.subheader("Detail Preprocessing")
                with st.expander("Lihat detail"):
                    st.write("Teks asli:", text_input)
                    st.write("Setelah cleaning:", re.sub(r'[^\w\s]', '', str(text_input).lower()))
                    st.write("Setelah stopword removal & stemming:", processed_text)

if __name__ == "__main__":
    # Download NLTK resources
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    main()
