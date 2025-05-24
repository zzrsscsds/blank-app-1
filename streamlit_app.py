import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import numpy as np

# Download NLTK resources
nltk_resources = [
    'punkt', 'stopwords', 'vader_lexicon', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'
]
for res in nltk_resources:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res)

st.set_page_config(page_title="Social Trends Forecaster", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #fafafa; }
        h1 { color: #1f77b4; }
        .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; }
        .stDownloadButton>button { background-color: #2ca02c; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Real-Time Social Media Trend Forecaster")

@st.cache_data
def load_combined_data():
    try:
        df = pd.read_csv("data/combined_social_data.csv")
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

sid = SentimentIntensityAnalyzer()

def compute_sentiment(text):
    try:
        return sid.polarity_scores(str(text))['compound']
    except:
        return 0.0

def extract_topics(texts):
    stop_words = set(stopwords.words('english'))
    processed_texts = [
        " ".join([
            word for word in doc.lower().split()
            if word.isalnum() and word not in stop_words
        ]) for doc in texts if isinstance(doc, str)
    ]
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(processed_texts)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(dtm)
    topics = lda.transform(dtm).argmax(axis=1)
    return topics
    
def drop_constant_columns(df):
    """Drop columns with only a single unique value to prevent VAR errors."""
    return df.loc[:, df.nunique() > 1]
combined_df = load_combined_data()
if combined_df.empty:
    st.stop()

st.sidebar.title("Filter Settings")
keyword = st.sidebar.text_input("Enter a topic keyword:", "#Fitness").lower().replace("#", "")

filtered_df = combined_df[combined_df['text'].str.lower().str.contains(keyword, na=False)].copy()

if not filtered_df.empty:
    filtered_df = filtered_df.dropna(subset=['text', 'created_at'])
    filtered_df['text'] = filtered_df['text'].astype(str)
    filtered_df['sentiment'] = filtered_df['text'].apply(compute_sentiment)
    filtered_df['engagement'] = pd.to_numeric(filtered_df.get('engagement', 0), errors='coerce').fillna(0)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['created_at'])
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    filtered_df['is_media'] = filtered_df['text'].str.contains('https://t.co', na=False).astype(int)
    filtered_df['topic'] = extract_topics(filtered_df['text'])
    filtered_df['text_length'] = filtered_df['text'].apply(len)
    filtered_df['hashtag_count'] = filtered_df['text'].apply(lambda x: x.count('#'))

    st.success(f"✅ Total filtered posts: {filtered_df.shape[0]}")

    time_df = filtered_df.groupby(filtered_df['timestamp'].dt.floor('h')).agg({
        'sentiment': 'mean',
        'engagement': 'sum',
        'topic': lambda x: x.mode()[0] if not x.mode().empty else 0,
        'hour': lambda x: x.mode()[0] if not x.mode().empty else 0,
        'is_media': 'mean'
    }).dropna()

    st.header("📈 Topic-Driven Engagement Forecasting")

    st.subheader("📌 Topic vs. Average Engagement")
    st.bar_chart(filtered_df.groupby('topic')['engagement'].mean())

    st.subheader("🌐 Word Cloud")
    stop_words = set(stopwords.words('english'))
    text = ' '.join(filtered_df['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader("⏰ Optimal Posting Times")
    hourly_engagement = filtered_df.groupby('hour')['engagement'].mean()
    fig, ax = plt.subplots()
    hourly_engagement.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Engagement')
    ax.set_title('Hourly Engagement')
    st.pyplot(fig)

    st.subheader("📅 Forecast with ARIMA")
    if len(time_df) >= 5:
        try:
            model = ARIMA(time_df['engagement'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=24)
            future_index = pd.date_range(start=time_df.index.max(), periods=24, freq='h')
            fig, ax = plt.subplots()
            ax.plot(future_index, forecast, label='Forecasted Engagement', color='tab:blue')
            ax.set_xlabel('Time')
            ax.set_ylabel('Engagement')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"ARIMA Forecast failed: {e}")

    st.subheader("🔮 Forecast with Prophet")
    try:
        prophet_df = time_df.reset_index().rename(columns={'timestamp': 'ds', 'engagement': 'y'})[['ds', 'y']]
        prophet_model = Prophet()
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=24, freq='h')
        forecast = prophet_model.predict(future)
        fig1 = prophet_model.plot(forecast)
        st.pyplot(fig1)
        fig2 = prophet_model.plot_components(forecast)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Prophet forecast error: {e}")

    st.subheader("🧠 Time Series Regression with VAR")
    try:
        model_data = time_df[['engagement', 'sentiment', 'topic', 'hour', 'is_media']]
        model = VAR(model_data)
        results = model.fit(maxlags=1)
        forecast = results.forecast(model_data.values[-1:], steps=24)
        forecast_df = pd.DataFrame(forecast, columns=model_data.columns)
        st.line_chart(forecast_df[['engagement']])
        with st.expander("Show VAR Coefficients"):
            st.dataframe(results.params)
    except Exception as e:
        st.warning(f"VAR model error: {e}")

    st.subheader("📈 Predict Engagement (Regression)")
    try:
        features = ['sentiment', 'text_length', 'hashtag_count', 'is_media']
        X = filtered_df[features].fillna(0)
        y = filtered_df['engagement'].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        chart_df = X_test.copy()
        chart_df['Predicted Engagement'] = y_pred
        chart_df['Actual Engagement'] = y_test.values

        fig, ax = plt.subplots()
        ax.plot(chart_df.index, chart_df['Predicted Engagement'], label='Predicted', color='tab:purple')
        ax.plot(chart_df.index, chart_df['Actual Engagement'], label='Actual', color='tab:red')
        ax.set_title('Predicted vs Actual Engagement')
        ax.set_xlabel('Post Index')
        ax.set_ylabel('Engagement')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Regression model error: {e}")

    st.download_button("📥 Download Data", filtered_df.to_csv(index=False), file_name="filtered_topic_data.csv")
else:
    st.warning("No posts found for this keyword.")


