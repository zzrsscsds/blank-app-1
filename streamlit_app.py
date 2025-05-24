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
        st.write(f"Loaded CSV with shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        st.error("Data file 'combined_social_data.csv' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# Compute sentiment using VADER
try:
    sid = SentimentIntensityAnalyzer()
except LookupError:
    st.error("Failed to load VADER lexicon. Please ensure 'vader_lexicon' is downloaded.")
    st.stop()

def compute_sentiment(text):
    try:
        return sid.polarity_scores(str(text))['compound']
    except:
        return 0.0

# Engagement prediction function
def preprocess_and_train(df):
    features = ['sentiment', 'text_length', 'hashtag_count', 'is_media']
    df['text_length'] = df['text'].apply(len)
    df['hashtag_count'] = df['text'].apply(lambda x: x.count('#'))
    df['is_media'] = df['text'].str.contains('https://t.co', na=False).astype(int)
    X = df[features].fillna(0)
    y = df['engagement'].fillna(0)
    if len(X) < 10:
        raise ValueError("Not enough data for training (minimum 10 rows required).")
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
    st.warning("No data loaded. Please check 'combined_social_data.csv'.")
    st.stop()

filtered_df = pd.DataFrame()

topic = st.text_input("Enter a topic keyword (e.g., #Fitness, Climate Change):", "#Fitness")

if topic:
    keyword = topic.replace("#", "").lower()
    topic_mask = combined_df['text'].str.lower().str.contains(keyword, na=False)
    filtered_df = combined_df.loc[topic_mask].copy()
    st.write(f"Filtered posts for '{keyword}': {filtered_df.shape[0]}")

    st.header(f"📱 Social Media Posts on {topic}")
    st.write(f"Total posts found: {filtered_df.shape[0]}")

    if not filtered_df.empty:
        # Drop rows with empty or NaN 'text', but preserve as much data as possible
        filtered_df = filtered_df.dropna(subset=['text'])
        filtered_df['text'] = filtered_df['text'].astype(str)

        # Compute sentiment if missing or invalid
        if 'sentiment' not in filtered_df.columns or filtered_df['sentiment'].isnull().all():
            filtered_df['sentiment'] = filtered_df['text'].apply(compute_sentiment)

        # Compute engagement if missing
        if 'engagement' not in filtered_df.columns:
            filtered_df['engagement'] = 0  # Placeholder; replace with actual logic if available

        st.write(f"Filtered DataFrame shape after dropping NaNs: {filtered_df.shape}")
        st.write("Sample texts:", filtered_df['text'].head(10).tolist())

        filtered_df['sentiment'] = pd.to_numeric(filtered_df['sentiment'], errors='coerce')
        filtered_df['engagement'] = pd.to_numeric(filtered_df['engagement'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=["created_at"])  # Only require valid 'created_at'

        # Engagement and Sentiment Over Time
        time_series = filtered_df.groupby(filtered_df['created_at'].dt.floor('H')).agg({
            'engagement': 'sum',
            'sentiment': 'mean'
        }).reset_index()

# 替换原始 chartjs 写法为原生 Streamlit 图表方式

# --- 时间趋势图 ---
        st.subheader("📊 Engagement & Sentiment Over Time")
        if not time_series.empty:
            fig, ax = plt.subplots()
            ax.plot(time_series['created_at'], time_series['engagement'], label='Engagement (Likes + Retweets)', color='tab:blue')
            ax.set_ylabel('Engagement', color='tab:blue')
            ax.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax.twinx()
            ax2.plot(time_series['created_at'], time_series['sentiment'], label='Sentiment (Mean)', color='tab:orange')
            ax2.set_ylabel('Sentiment', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            ax.set_xlabel('Time')
            ax.set_title('Engagement & Sentiment Over Time')
            fig.autofmt_xdate()
            st.pyplot(fig)
        else:
            st.warning("No data available for engagement and sentiment trends. Check if 'created_at' is valid.")

        # --- 情绪分布柱状图 ---
        st.subheader("💬 Sentiment Distribution")
        if filtered_df['sentiment'].notnull().sum() > 0:
            sentiment_binned = pd.cut(filtered_df['sentiment'], bins=10)
            sentiment_counts = sentiment_binned.value_counts().sort_index()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color='tab:green')
            ax.set_title('Sentiment Distribution')
            ax.set_xlabel('Sentiment Range')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.warning("No sentiment data available to display distribution.")

        # --- 最佳发布时间 ---
        st.subheader("⏰ Optimal Posting Times")
        filtered_df['hour'] = filtered_df['created_at'].dt.hour
        hourly_engagement = filtered_df.groupby('hour')['engagement'].mean().reset_index()
        if not hourly_engagement.empty:
            fig, ax = plt.subplots()
            ax.bar(hourly_engagement['hour'], hourly_engagement['engagement'], color='mediumpurple')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Average Engagement')
            ax.set_title('Optimal Posting Times')
            st.pyplot(fig)
        else:
            st.warning("No data available for optimal posting times.")

        # --- 预测结果对比图 ---
        st.subheader("📈 Predicted vs Actual Engagement")
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

        if len(processed_texts) >= 2:
            vectorizer = CountVectorizer(max_df=0.9, min_df=1, max_features=1000)
            dtm = vectorizer.fit_transform(processed_texts)
            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            lda.fit(dtm)
            words = vectorizer.get_feature_names_out()
            for i, topic_dist in enumerate(lda.components_):
                topic_words = [words[i] for i in topic_dist.argsort()[-5:][::-1]]
                st.write(f"**Topic {i+1}:** {', '.join(topic_words)}")
        else:
            hashtags = filtered_df['text'].str.findall(r'#\\w+').explode().value_counts().head(5)
            st.write("**Top Hashtags**: " + ", ".join(hashtags.index if not hashtags.empty else ["No hashtags found"]))


        st.subheader("📅 48-Hour Engagement Forecast")
        if len(time_series) >= 5:
            try:
                model = ARIMA(time_series['engagement'].fillna(0), order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=48)
                forecast_df = pd.DataFrame({
                    'Time': pd.date_range(start=time_series['created_at'].max(), periods=48, freq='H'),
                    'Forecasted Engagement': forecast
                })
                chart_config = {
                    "type": "line",
                    "data": {
                        "labels": forecast_df['Time'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
                        "datasets": [{
                            "label": "Forecasted Engagement",
                            "data": forecast_df['Forecasted Engagement'].tolist(),
                            "borderColor": "#2ca02c",
                            "backgroundColor": "rgba(44, 160, 44, 0.2)",
                            "fill": False
                        }]
                    },
                    "options": {
                        "scales": {
                            "y": {"title": {"display": True, "text": "Engagement"}},
                            "x": {"title": {"display": True, "text": "Time"}}
                        },
                        "plugins": {
                            "legend": {"display": True},
                            "title": {"display": True, "text": "48-Hour Engagement Forecast"}
                        }
                    }
                }
                st.write("```chartjs\n" + str(chart_config) + "\n```")
            except Exception as e:
                st.warning(f"Failed to generate forecast: {e}")
        else:
            st.warning("Not enough data for trend forecasting.")

        st.subheader("⏰ Optimal Posting Times")
        filtered_df['hour'] = filtered_df['created_at'].dt.hour
        hourly_engagement = filtered_df.groupby('hour')['engagement'].mean().reset_index()
        if not hourly_engagement.empty:
            chart_config = {
                "type": "bar",
                "data": {
                    "labels": hourly_engagement['hour'].astype(str).tolist(),
                    "datasets": [{
                        "label": "Average Engagement",
                        "data": hourly_engagement['engagement'].tolist(),
                        "backgroundColor": "#9467bd",
                        "borderColor": "#7f2a9c",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {"title": {"display": True, "text": "Average Engagement"}},
                        "x": {"title": {"display": True, "text": "Hour of Day"}}
                    },
                    "plugins": {
                        "legend": {"display": False},
                        "title": {"display": True, "text": "Optimal Posting Times"}
                    }
                }
            }
            st.write("```chartjs\n" + str(chart_config) + "\n```")
        else:
            st.warning("No data available for optimal posting times.")

         st.subheader("📈 Predicting Engagement")

        try:
            with st.spinner("Training ML model..."):
                result = preprocess_and_train(filtered_df)
                
            st.success("✅ Model trained successfully!")
            X_test = result['X_test']
            y_test = result['y_test']
            y_pred = result['y_pred']

            st.write(f"**R² Score**: {result['r2_score']:.2f}")
            st.write(f"**RMSE**: {result['rmse']:.2f}")

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
            st.error(f"❌ Model training or plotting error: {e}")

# Prophet Forecasting on Full Dataset (Global Time Series)
st.subheader("📅 48-Hour Engagement Forecast (Global)")
try:
    from prophet import Prophet

    global_ts = combined_df.copy()
    global_ts['created_at'] = pd.to_datetime(global_ts['created_at'], errors='coerce')
    global_ts = global_ts.dropna(subset=['created_at', 'engagement'])
    global_ts = global_ts.groupby(global_ts['created_at'].dt.floor('h')).agg({
        'engagement': 'sum'
    }).reset_index()

    if len(global_ts) >= 20:
        df_prophet = global_ts.rename(columns={
            'created_at': 'ds',
            'engagement': 'y'
        })

        prophet_model = Prophet()
        prophet_model.fit(df_prophet)

        future = prophet_model.make_future_dataframe(periods=48, freq='h')
        forecast = prophet_model.predict(future)

        st.subheader("🔮 Global Forecast of Engagement (Next 48 Hours)")
        fig1 = prophet_model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("📈 Components of Global Trend")
        fig2 = prophet_model.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.warning("📉 Not enough global time series data for Prophet prediction. Try with more complete dataset.")

except ImportError:
    st.error("Prophet library not found. Please run: pip install prophet")
except Exception as e:
    st.error(f"Prophet model error: {e}")

# Enhanced Regression Features for Engagement Prediction
with st.spinner("Training ML model..."):
    result = preprocess_and_train(filtered_df)
X_test = result['X_test']
y_test = result['y_test']
y_pred = result['y_pred']

st.subheader("📈 Predicting Engagement (Enhanced Features)")
filtered_df = filtered_df.rename(columns={"created_at": "timestamp"})
try:
    with st.spinner("Training enhanced regression model..."):
        filtered_df['hour'] = filtered_df['timestamp'].dt.hour
        filtered_df['is_weekend'] = filtered_df['timestamp'].dt.dayofweek >= 5
        filtered_df['text_length'] = filtered_df['text'].apply(len)
        filtered_df['hashtag_count'] = filtered_df['text'].apply(lambda x: x.count('#'))
        filtered_df['is_media'] = filtered_df['text'].str.contains('https://t.co', na=False).astype(int)

        features = ['sentiment', 'text_length', 'hashtag_count', 'is_media', 'hour', 'is_weekend']
        X = filtered_df[features].fillna(0)
        y = filtered_df['engagement'].fillna(0)

        if len(X) < 10:
            raise ValueError("Not enough data for regression training (min 10 rows required).")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Enhanced Model trained successfully!")
        st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")
        st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        chart_df = X_test.copy()
        chart_df['Predicted Engagement'] = y_pred
        chart_df['Actual Engagement'] = y_test.values

        st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])

except Exception as e:
    st.error(f"❌ Enhanced model error: {e}")
