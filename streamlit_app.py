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

st.write("CSV 文件加载成功：", combined_df.shape)

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

        st.subheader("📊 Engagement & Sentiment Over Time")
        if not time_series.empty:
            labels = time_series['created_at'].dt.strftime('%Y-%m-%d %H:%M').tolist()
            engagement_data = time_series['engagement'].fillna(0).tolist()
            sentiment_data = time_series['sentiment'].fillna(0).tolist()
            chart_config = {
                "type": "line",
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": "Engagement (Likes + Retweets)",
                            "data": engagement_data,
                            "borderColor": "#1f77b4",
                            "backgroundColor": "rgba(31, 119, 180, 0.2)",
                            "fill": False,
                            "yAxisID": "y"
                        },
                        {
                            "label": "Sentiment (Mean)",
                            "data": sentiment_data,
                            "borderColor": "#ff7f0e",
                            "backgroundColor": "rgba(255, 127, 14, 0.2)",
                            "fill": False,
                            "yAxisID": "y1"
                        }
                    ]
                },
                "options": {
                    "scales": {
                        "y": {
                            "type": "linear",
                            "display": True,
                            "position": "left",
                            "title": {"display": True, "text": "Engagement"}
                        },
                        "y1": {
                            "type": "linear",
                            "display": True,
                            "position": "right",
                            "title": {"display": True, "text": "Sentiment"},
                            "grid": {"drawOnChartArea": False}
                        },
                        "x": {
                            "title": {"display": True, "text": "Time"}
                        }
                    },
                    "plugins": {
                        "legend": {"display": True},
                        "title": {"display": True, "text": "Engagement & Sentiment Over Time"}
                    }
                }
            }
            st.write("```chartjs\n" + str(chart_config) + "\n```")
        else:
            st.warning("No data available for engagement and sentiment trends. Check if 'created_at' is valid.")

        st.subheader("💬 Sentiment Distribution")
        if filtered_df['sentiment'].notnull().sum() > 0:
            sentiment_binned = pd.cut(filtered_df['sentiment'], bins=10)
            sentiment_counts = sentiment_binned.value_counts().sort_index()
            chart_config = {
                "type": "bar",
                "data": {
                    "labels": [str(interval) for interval in sentiment_counts.index],
                    "datasets": [{
                        "label": "Sentiment Distribution",
                        "data": sentiment_counts.tolist(),
                        "backgroundColor": "#2ca02c",
                        "borderColor": "#1f77b4",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {"title": {"display": True, "text": "Count"}},
                        "x": {"title": {"display": True, "text": "Sentiment Range"}}
                    },
                    "plugins": {
                        "legend": {"display": False},
                        "title": {"display": True, "text": "Sentiment Distribution"}
                    }
                }
            }
            st.write("```chartjs\n" + str(chart_config) + "\n```")
        else:
            st.warning("No sentiment data available to display distribution.")

        st.subheader("🔍 Sample Posts")
        st.dataframe(filtered_df[['created_at', 'text', 'sentiment', 'engagement']].head(10))

        st.subheader("🌐 Word Cloud")
        text = " ".join(filtered_df['text'].tolist())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No text available for word cloud. Using sample text.")
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate("fitness gym workout motivation health")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        st.subheader("🔍 Popular Subtopics within this Topic")
        stop_words = set(stopwords.words('english')) - {'run', 'pump'}
        texts = filtered_df['text'].tolist()
        st.write(f"Number of posts after filtering: {len(texts)}")

        processed_texts = []
        for doc in texts:
            tokens = [
                word for word in word_tokenize(doc.lower())
                if (word.isalnum() or word.startswith('#') or word in ['🦵🏽', '💪🏽']) and word not in stop_words
            ]
            processed_texts.append(" ".join(tokens))
        processed_texts = [doc for doc in processed_texts if len(doc.split()) > 1]
        st.write(f"Number of posts after cleaning: {len(processed_texts)}")

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
            hashtags = filtered_df['text'].str.findall(r'#\w+').explode().value_counts().head(5)
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
        filtered_df = filtered_df.rename(columns={"created_at": "timestamp"})
        try:
            with st.spinner("Training ML model..."):
                result = preprocess_and_train(filtered_df)
            st.success("✅ Model trained successfully!")
            st.write(f"**R² Score**: {result['r2_score']:.2f}")
            st.write(f"**RMSE**: {result['rmse']:.2f}")
            chart_df = result['X_test'].copy()
            chart_df['Predicted Engagement'] = result['y_pred']
            chart_df['Actual Engagement'] = result['y_test'].values
            chart_config = {
                "type": "line",
                "data": {
                    "labels": chart_df.index.astype(str).tolist(),
                    "datasets": [
                        {
                            "label": "Predicted Engagement",
                            "data": chart_df['Predicted Engagement'].tolist(),
                            "borderColor": "#9467bd",
                            "backgroundColor": "rgba(148, 103, 189, 0.2)",
                            "fill": False
                        },
                        {
                            "label": "Actual Engagement",
                            "data": chart_df['Actual Engagement'].tolist(),
                            "borderColor": "#d62728",
                            "backgroundColor": "rgba(214, 39, 40, 0.2)",
                            "fill": False
                        }
                    ]
                },
                "options": {
                    "scales": {
                        "y": {"title": {"display": True, "text": "Engagement"}},
                        "x": {"title": {"display": True, "text": "Post Index"}}
                    },
                    "plugins": {
                        "legend": {"display": True},
                        "title": {"display": True, "text": "Predicted vs Actual Engagement"}
                    }
                }
            }
            st.write("```chartjs\n" + str(chart_config) + "\n```")
        except Exception as e:
            st.error(f"❌ Model error: {e}")

    else:
        st.warning(f"No posts found for '{topic}'. Check if the keyword exists in the data.")
