import streamlit as st
import nltk

@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk()
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Setup ---
st.set_page_config(page_title="Spam Classifier", page_icon="📩")


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Text Preprocessing ---
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    words = [ps.stem(w) for w in words]

    return " ".join(words)

# --- Load Model ---
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('NLP_model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

# --- UI ---
st.title("📩 Email / SMS Spam Classifier")
st.markdown("Detect whether a message is **Spam 🚫** or **Not Spam ✅**")

input_text = st.text_area("✍️ Enter your message here")

# --- Prediction ---
if st.button("🚀 Predict"):

    if input_text.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Preprocess
        transformed = transform_text(input_text)

        # Vectorize (IMPORTANT FIX)
        vector_input = tfidf.transform([transformed]).toarray()

        # Prediction
        proba = model.predict_proba(vector_input)[0][1]

        # --- Output ---
        if proba >= 0.4:
            st.error("🚫 Spam Detected")
        else:
            st.success("✅ Not Spam")

        # --- Confidence Score ---
        st.subheader("📊 Confidence Score")
        st.progress(int(proba * 100))

        st.write(f"Spam Probability: **{proba:.2f}**")