import pandas as pd
import numpy as np
import os
import re
import string
import nltk
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from io import BytesIO
import base64
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Function to download required NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
        except LookupError:
            nltk.download(res)

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Train model
def train_model():
    df = pd.read_csv('fake_and_real_news.csv')
    df['processed_text'] = df['Text'].apply(preprocess_text)
    df['label_binary'] = df['label'].map({'Real': 0, 'Fake': 1})
    
    tfidf = TfidfVectorizer(max_features=10000)
    X = tfidf.fit_transform(df['processed_text'])
    y = df['label_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], output_dict=True)

    return model, tfidf, accuracy, report, df

# Load model
def load_model():
    model = joblib.load('fake_news_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

# Predict
def predict_news(text, model, tfidf):
    processed = preprocess_text(text)
    tfidf_input = tfidf.transform([processed])
    prediction = model.predict(tfidf_input)[0]
    confidence = model.predict_proba(tfidf_input)[0][prediction] * 100
    return ('Fake' if prediction == 1 else 'Real'), confidence

# Word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

# Top TF-IDF words
def get_top_tfidf_features(tfidf, df, label, top_n=10):
    subset = df[df['label'] == label]
    tfidf_matrix = tfidf.transform(subset['processed_text'])
    feature_names = tfidf.get_feature_names_out()
    mean_scores = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_scores.argsort()[-top_n:][::-1]
    return [(feature_names[i], mean_scores[i]) for i in top_indices]

# Main App
def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict", "Visualizations"])

    download_nltk_resources()

    try:
        model, tfidf = load_model()
        df = pd.read_csv('fake_and_real_news.csv')
        df['processed_text'] = df['Text'].apply(preprocess_text)
        df['label_binary'] = df['label'].map({'Real': 0, 'Fake': 1})
        accuracy, report = None, None
    except:
        st.info("Training model... Please wait.")
        model, tfidf, accuracy, report, df = train_model()
        st.success("Model trained!")

    if page == "Predict":
        st.title("📰 Fake News Detector")
        user_input = st.text_area("Enter news text to classify:", height=200)
        if st.button("Predict"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    label, confidence = predict_news(user_input, model, tfidf)
                    st.subheader("Prediction")
                    st.success(f"This news is **{label}** with {confidence:.2f}% confidence")
            else:
                st.warning("Please enter some text.")

    elif page == "Visualizations":
        st.title("📊 Dataset Insights")

        st.subheader("Label Distribution")
        label_dist = df['label'].value_counts().reset_index()
        label_dist.columns = ['Label', 'Count']
        st.plotly_chart(px.pie(label_dist, names='Label', values='Count'))

        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)
        with col1:
            fake_text = ' '.join(df[df['label'] == 'Fake']['processed_text'])
            fake_wc = generate_word_cloud(fake_text, "Fake News")
            st.image(f"data:image/png;base64,{fake_wc}", caption="Fake News Word Cloud")
        with col2:
            real_text = ' '.join(df[df['label'] == 'Real']['processed_text'])
            real_wc = generate_word_cloud(real_text, "Real News")
            st.image(f"data:image/png;base64,{real_wc}", caption="Real News Word Cloud")

        st.subheader("Top TF-IDF Features")
        col1, col2 = st.columns(2)
        with col1:
            top_fake = get_top_tfidf_features(tfidf, df, 'Fake')
            df_fake = pd.DataFrame(top_fake, columns=["Word", "TF-IDF Score"])
            st.plotly_chart(px.bar(df_fake, x='Word', y='TF-IDF Score', title="Top Fake News Words"))
        with col2:
            top_real = get_top_tfidf_features(tfidf, df, 'Real')
            df_real = pd.DataFrame(top_real, columns=["Word", "TF-IDF Score"])
            st.plotly_chart(px.bar(df_real, x='Word', y='TF-IDF Score', title="Top Real News Words"))

        if report:
            st.subheader("Classification Report")
            metrics = ['precision', 'recall', 'f1-score']
            real_metrics = [report['Real'][m] for m in metrics]
            fake_metrics = [report['Fake'][m] for m in metrics]
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Real', x=metrics, y=real_metrics, marker_color='green'))
            fig.add_trace(go.Bar(name='Fake', x=metrics, y=fake_metrics, marker_color='red'))
            fig.update_layout(barmode='group', title="Model Performance by Class")
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
