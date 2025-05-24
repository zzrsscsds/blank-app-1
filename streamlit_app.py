import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(layout="wide")
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

try:
    sid = SentimentIntensityAnalyzer()
except LookupError:
    st.error("Failed to load VADER lexicon.")
    st.stop()

def compute_sentiment(text):
    try:
        return sid.polarity_scores(str(text))['compound']
    except:
        return 0.0

def preprocess_and_train(df):
    features = ['sentiment', 'text_length', 'hashtag_count', 'is_media']
    df['text_length'] = df['text'].apply(len)
    df['hashtag_count'] = df['text'].apply(lambda x: x.count('#'))
    df['is_media'] = df['text'].str.contains('https://t.co', na=False).astype(int)
    X = df[features].fillna(0)
    y = df['engagement'].fillna(0)
    if len(X) < 10:
        raise ValueError("Not enough data for training.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'r2_score': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

combined_df = load_combined_data()
if combined_df.empty:
    st.stop()

st.sidebar.title("Keyword Filter")
topic = st.sidebar.text_input("Enter a topic keyword:", "#Fitness")

if topic:
    keyword = topic.lower().replace("#", "")
    mask = combined_df['text'].str.lower().str.contains(keyword, na=False)
    filtered_df = combined_df[mask].copy()

    if filtered_df.empty:
        st.warning("No posts found for this topic.")
        st.stop()

    filtered_df = filtered_df.dropna(subset=['text', 'created_at'])
    filtered_df['text'] = filtered_df['text'].astype(str)
    filtered_df['created_at'] = pd.to_datetime(filtered_df['created_at'])
    filtered_df['sentiment'] = filtered_df['text'].apply(compute_sentiment)
    filtered_df['engagement'] = pd.to_numeric(filtered_df['engagement'], errors='coerce').fillna(0)

    st.subheader("📄 Sample Posts")
    st.dataframe(filtered_df[['created_at', 'text', 'engagement', 'sentiment']].head(10))

    st.subheader("📊 Engagement & Sentiment Over Time")
    time_series = filtered_df.groupby(filtered_df['created_at'].dt.floor('h')).agg({
        'engagement': 'sum',
        'sentiment': 'mean'
    }).reset_index()

    if not time_series.empty:
        fig, ax1 = plt.subplots()
        ax1.plot(time_series['created_at'], time_series['engagement'], color='tab:blue')
        ax1.set_ylabel('Engagement', color='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(time_series['created_at'], time_series['sentiment'], color='tab:orange')
        ax2.set_ylabel('Sentiment', color='tab:orange')
        fig.autofmt_xdate()
        st.pyplot(fig)

    st.subheader("💬 Sentiment Distribution")
    binned = pd.cut(filtered_df['sentiment'], bins=10)
    counts = binned.value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax, color='tab:green')
    ax.set_xlabel('Sentiment Range')
    ax.set_ylabel('Post Count')
    st.pyplot(fig)

    st.subheader("🌐 Word Cloud")
    stop_words = set(stopwords.words('english'))
    text = ' '.join(filtered_df['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader("📅 Optimal Posting Time")
    filtered_df['hour'] = filtered_df['created_at'].dt.hour
    hourly_engagement = filtered_df.groupby('hour')['engagement'].mean()
    fig, ax = plt.subplots()
    hourly_engagement.plot(kind='bar', ax=ax, color='mediumpurple')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Engagement')
    st.pyplot(fig)

    st.subheader("🧠 Topic Modeling")
    processed_texts = [
        " ".join([
            word for word in word_tokenize(doc.lower())
            if word.isalnum() and word not in stop_words
        ]) for doc in filtered_df['text']
    ]
    processed_texts = [doc for doc in processed_texts if len(doc.split()) > 1]

    if len(processed_texts) >= 2:
        vectorizer = CountVectorizer(max_df=0.9, min_df=1, max_features=1000)
        dtm = vectorizer.fit_transform(processed_texts)
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)
        words = vectorizer.get_feature_names_out()
        for i, topic_dist in enumerate(lda.components_):
            top_words = [words[i] for i in topic_dist.argsort()[-5:][::-1]]
            st.markdown(f"**Topic {i+1}:** {' | '.join(top_words)}")

    st.subheader("🔮 48-Hour Forecast (ARIMA)")
    if len(time_series) >= 5:
        try:
            model = ARIMA(time_series['engagement'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=48)
            future_index = pd.date_range(start=time_series['created_at'].max(), periods=48, freq='h')
            fig, ax = plt.subplots()
            ax.plot(future_index, forecast, label='Forecasted Engagement', color='tab:blue')
            ax.set_xlabel('Time')
            ax.set_ylabel('Forecasted Engagement')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"ARIMA Forecast failed: {e}")

    st.subheader("📈 Predicting Engagement")
    try:
        result = preprocess_and_train(filtered_df)
        st.success("✅ Model trained successfully!")
        st.write(f"**R² Score**: {result['r2_score']:.2f}")
        st.write(f"**RMSE**: {result['rmse']:.2f}")
        chart_df = result['X_test'].copy()
        chart_df['Predicted Engagement'] = result['y_pred']
        chart_df['Actual Engagement'] = result['y_test'].values
        fig, ax = plt.subplots()
        ax.plot(chart_df.index, chart_df['Predicted Engagement'], label='Predicted', color='tab:purple')
        ax.plot(chart_df.index, chart_df['Actual Engagement'], label='Actual', color='tab:red')
        ax.set_title('Predicted vs Actual Engagement')
        ax.set_xlabel('Index')
        ax.set_ylabel('Engagement')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Model prediction error: {e}")

    try:
        from prophet import Prophet
        st.subheader("📅 Prophet Forecast (Global)")
        df_prophet = combined_df[['created_at', 'engagement']].dropna().copy()
        df_prophet = df_prophet.rename(columns={'created_at': 'ds', 'engagement': 'y'})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=48, freq='h')
        forecast = model.predict(future)
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Prophet model failed: {e}")


