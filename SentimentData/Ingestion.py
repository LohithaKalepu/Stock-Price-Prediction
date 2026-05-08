import pandas as pd
import requests
from dotenv import load_dotenv
from newspaper import Article, Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
import os
import datetime
import time

load_dotenv()

#--Configuration--
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
if not MARKETAUX_API_KEY:
    raise ValueError("FATAL ERROR: MARKETAUX_API_KEY not found in environment variables!")
BASE_URL='https://api.marketaux.com/v1/news/all'
CSV_FIle = 'stock_sentiment_history.csv'
TARGET_TICKER = "GOOGL"
COMPANY_NAME = "Google"
blacklist = "fortune.com,bloomberg.com,wsj.com,barrons.com,ft.com,nytimes.com,seekingalpha.com,marketwatch.com,gurufocus.com,investing.com,thestreet.com,dailyfx.com,goodreturns.in"

#Newspaper3k Config
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
config.request_timeout = 15

#NLP & Sentiment Models
nlp = spacy.load("en_core_web_sm")

# Finbert setup
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def get_daily_sentiment(target_date):
    """Fetches articles for a specific date and returns an aggregated score."""
    all_articles = []
    for page_num in range(1, 3):
        params = {'api_token': MARKETAUX_API_KEY,
            'symbols':TARGET_TICKER,
            'limit':3,
            'page': page_num,
            'exclude_domains': blacklist,
            'filter_entities': 'true',
            'published_on': target_date,
            'language': 'en'
            }
        
        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return None
        
        data = response.json()
        
        
        

        if 'data' in data:
            for art in data['data']:
                    
                    scores = []
                    relevant_sentences = []
                    raw_url = art['url']
                    clean_url = raw_url.strip().replace("'n", "").replace("\\n", "")

                    try:
                        scraper = Article(clean_url, config=config)
                        scraper.download()
                        scraper.parse()
                        print(f"Scraped {len(scraper.text)} characters from {clean_url}")

                        if len(scraper.text) > 100:
                            doc = nlp(scraper.text)
                            sents = list(doc.sents)
                            unique_contexts =[]
                            seen_indicies = set()
                            #want to get sentence before and after
                            for i, s in enumerate(sents):
                                if COMPANY_NAME.lower() in s.text.lower() and i not in seen_indicies:
                                    window = sents[max(0,i-1) : min(len(sents), i + 2)]
                                    unique_contexts.append(" ".join([i.text.strip() for i in window]))

                                    # mark indicies as seen
                                    for j in range(max(0,i-1), min(len(sents),i+2)):
                                        seen_indicies.add(j)

                            if unique_contexts:
                                sentiments = classifier(unique_contexts[:10])

                                #calculate scores Pos, Neg, Neutral
                                for s in sentiments:
                                    if s['label'] == 'positive':
                                        scores.append(s['score'])
                                    elif s['label'] == 'negative':
                                        scores.append(-s['score'])
                                    else:
                                        scores.append(0) #Netural counts as 0
                                    
                            
                        
                        if scores:
                            article_avg = sum(scores) / len(scores)
                            all_articles.append(article_avg)


                    except Exception as e:
                        print(f"Failed to scrape {clean_url}: {e}")

    if all_articles:
         final_daily_score = sum(all_articles) / len(all_articles)

         return {
                'Date': target_date,
                'Ticker': TARGET_TICKER,
                'Final_Sentiment': round(final_daily_score, 4),
                'Article_Count': len(all_articles)
         }
    return None

start_date = datetime.datetime(2022, 4, 1)
end_date = datetime.datetime(2022, 5, 1)

current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"--- Fetching: {date_str} ---")

    day_data = get_daily_sentiment(date_str)

    if day_data:
          df = pd.DataFrame([day_data])
          df.to_csv(CSV_FIle, mode='a', index=False, header=not os.path.exists(CSV_FIle))
    else:
        print(f"no data Found  for {date_str} (Day_data was None)")
        gap_data = {
            'Date': date_str,
            'Ticker': TARGET_TICKER,
            'Final_Sentiment': 0.0,
            'Article_Count': 0
        }
        df = pd.DataFrame([gap_data])
        df.to_csv(CSV_FIle, mode='a', index=False, header=not os.path.exists(CSV_FIle))

    time.sleep(1.5)
    current_date += datetime.timedelta(days=1)

print(f"{start_date} to {end_date} complete and saved to {CSV_FIle}")