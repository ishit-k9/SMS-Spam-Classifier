import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import time

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load your trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_detector.pkl', 'rb'))

# Define text transformation function (same as your training pipeline)
def transform(text):
    text = text.lower()
    text = word_tokenize(text)
    
    new_text = []
    for word in text:
        if word.isalnum():
            new_text.append(word)
    
    text = new_text[:]
    new_text.clear()
    
    for word in text:
        if word not in string.punctuation and word not in stopwords.words('english'):
            new_text.append(word)
    
    text = new_text[:]
    new_text.clear()
    
    ps = PorterStemmer()
    for word in text:
        new_text.append(ps.stem(word))
        
    return " ".join(new_text)

# Configure page settings
st.set_page_config(
    page_title="SMS Spam Guardian",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    .stTextArea>div>div>textarea {
        background-color: #FFFFFF;
        color: #2C3E50;
    }
    .stButton>button {
        background-color: #2980B9;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #3498DB;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>📱 SMS Spam Guardian</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #34495E; margin-bottom: 30px;'>Detect Spam Messages with AI</h3>", 
            unsafe_allow_html=True)

# Input Section
with st.container():
    message = st.text_area("Enter your message here:", height=150, 
                         placeholder="Paste your SMS text here...")

# Prediction Section
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button("Check for Spam 🕵️♂️")

if predict_button:
    with st.spinner('Analyzing message...'):
        # Preprocess the text
        transformed_text = transform(message)
        
        # Vectorize the text
        vector_input = tfidf.transform([transformed_text])
        
        # Make prediction
        prediction = model.predict(vector_input)[0]
        
        # Add some delay for better UX
        time.sleep(1)
        
        # Show result with animation
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box" style="background-color: #FDEDEC; color: #C0392B; border: 2px solid #C0392B;">
                🚨 Spam Alert! This message looks suspicious!
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box" style="background-color: #EAFAF1; color: #28B463; border: 2px solid #28B463;">
                ✅ Safe Message! This looks like a genuine message!
                </div>
            """, unsafe_allow_html=True)

# Explanation Section
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    This spam detection system uses:
    - **Natural Language Processing (NLP)** to understand message content
    - **Machine Learning** (Random Forest algorithm) to identify spam patterns
    - Advanced text preprocessing including:
        * Lowercasing
        * Tokenization
        * Stopword removal
        * Stemming
    - TF-IDF vectorization for text representation
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #7F8C8D;'>🔒 Your messages are processed securely and never stored</div>", 
            unsafe_allow_html=True)