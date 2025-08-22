"""
News scraper for finance
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article, Config
from finvader import finvader
from googlenewsdecoder import new_decoderv1
from dateutil import parser
import datetime as dt
import random
import os
from urllib.parse import urlparse


class NewsScaper:
    def __init__(self):
        """initialize the scraper with configuration"""
        # user agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0'
        ]
        
        # setup newspaper config
        self.config = Config()
        self.config.browser_user_agent = random.choice(self.user_agents)
        self.config.request_timeout = 15
        self.config.enable_image_fetching = False
        
        # setup sentiment analyzer for financial content
        from finvader import finvader
        
        # create output directory
        os.makedirs('output', exist_ok=True)
        
    def load_sources(self, filepath='source_list.csv'):
        """load approved news sources from csv"""
        try:
            sources_df = pd.read_csv(filepath)
            # expected columns: source_name, source_domain
            return set(sources_df['source_name'].str.lower().str.strip())
        except FileNotFoundError:
            print(f"ERROR: {filepath} not found!")
            return set()
    
    def load_search_terms(self, filepath='search_terms.csv'):
        """load search terms from csv"""
        try:
            terms_df = pd.read_csv(filepath)
            # expected columns: topic_id, search_term
            return terms_df.to_dict('records')
        except FileNotFoundError:
            print(f"ERROR: {filepath} not found!")
            return []
    
    def get_existing_links(self, filepath='output/news_results.csv'):
        """get already processed article links to avoid duplicates"""
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            return set(existing_df['link'].dropna().str.lower())
        return set()
    
    def search_google_news(self, search_term, days=7):
        """search google news for articles"""
        url = f"https://news.google.com/rss/search?q={search_term}%20when%3A{days}d"
        headers = {'User-Agent': random.choice(self.user_agents)}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'xml')
            
            articles = []
            for item in soup.find_all("item"):
                article_data = {
                    'title': item.title.text.strip(),
                    'encoded_url': item.link.text.strip(),
                    'source': item.source.text.strip().lower(),
                    'pub_date': item.pubDate.text if item.pubDate else None
                }
                articles.append(article_data)
            
            return articles
        except Exception as e:
            print(f"Error searching for '{search_term}': {e}")
            return []
    
    def decode_url(self, encoded_url):
        """decode google news url"""
        try:
            result = new_decoderv1(encoded_url, interval=3)
            if result.get("status"):
                return result['decoded_url'].strip()
        except Exception as e:
            print(f"URL decode error: {e}")
        return None
    
    def is_valid_source(self, source_name, domain_url, approved_sources):
        """check if source is in approved list and meets criteria"""
        # check approved sources
        if source_name not in approved_sources:
            return False
        
        # check domain extension
        parsed = urlparse(domain_url)
        domain = parsed.netloc.lower()
        valid_extensions = ('.com', '.org', '.net', '.edu')
        
        return any(domain.endswith(ext) for ext in valid_extensions)
    
    def extract_article_content(self, url):
        """extract article content and perform sentiment analysis"""
        try:
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            article.nlp()
            
            # get content
            summary = article.summary or article.text or ""
            keywords = article.keywords or []
            
            # skip very short articles
            if len(summary) < 100:
                return None
            
            # perform sentiment analysis using finvader (optimized for financial content)
            compound_score = finvader(
                summary,
                use_sentibignomics=True,  # financial lexicon 1
                use_henry=True,           # financial lexicon 2  
                indicator="compound"      # get compound score
            )
            
            if compound_score <= -0.05:
                sentiment = 'negative'
            elif compound_score >= 0.05:
                sentiment = 'positive'
            else:
                sentiment = 'neutral'
            
            return {
                'summary': summary,
                'keywords': keywords,
                'sentiment': sentiment,
                'polarity': compound_score
            }
            
        except Exception as e:
            print(f"Error processing article {url}: {e}")
            return None
    
    def get_next_archive_number(self):
        """get the next available archive number"""
        archive_files = [f for f in os.listdir('output') if f.startswith('news_archive_') and f.endswith('.csv')]
        if not archive_files:
            return 1
        
        # extract numbers from existing archive files
        numbers = []
        for filename in archive_files:
            try:
                # extract number from 'news_archive_X.csv'
                num = int(filename.replace('news_archive_', '').replace('.csv', ''))
                numbers.append(num)
            except ValueError:
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def archive_old_data(self, df, cutoff_date):
        """archive data older than cutoff date"""
        old_data = df[df['published_date'] < cutoff_date].copy()
        
        if not old_data.empty:
            archive_num = self.get_next_archive_number()
            archive_path = f'output/news_archive_{archive_num}.csv'
            old_data.to_csv(archive_path, index=False)
            print(f"Archived {len(old_data)} old articles to {archive_path}")
        
        # return only current data
        return df[df['published_date'] >= cutoff_date].copy()
    
    def run_scraper(self):
        """main scraper function"""
        print("Starting news scraper...")
        
        # load configuration
        approved_sources = self.load_sources()
        search_terms = self.load_search_terms()
        existing_links = self.get_existing_links()
        
        if not approved_sources:
            print("No approved sources found. Exiting.")
            return
        
        if not search_terms:
            print("No search terms found. Exiting.")
            return
        
        print(f"Loaded {len(approved_sources)} approved sources")
        print(f"Loaded {len(search_terms)} search terms")
        print(f"Found {len(existing_links)} existing articles")
        
        # collect articles
        all_articles = []
        
        for term_data in search_terms:
            topic_id = term_data.get('topic_id')
            search_term = term_data.get('search_term')
            
            print(f"\nSearching for: {search_term}")
            
            articles = self.search_google_news(search_term)
            processed_count = 0
            
            for article in articles:
                # decode url
                decoded_url = self.decode_url(article['encoded_url'])
                if not decoded_url:
                    continue
                
                # skip if already processed
                if decoded_url.lower() in existing_links:
                    continue
                
                # check if source is approved
                if not self.is_valid_source(article['source'], decoded_url, approved_sources):
                    continue
                
                # extract content
                content = self.extract_article_content(decoded_url)
                if not content:
                    continue
                
                # parse date
                try:
                    pub_date = parser.parse(article['pub_date']).date()
                except:
                    pub_date = dt.date.today()
                
                # compile article data
                article_record = {
                    'topic_id': topic_id,
                    'search_term': search_term,
                    'title': article['title'],
                    'summary': content['summary'],
                    'keywords': ', '.join(content['keywords']),
                    'published_date': pub_date,
                    'link': decoded_url,
                    'source': article['source'],
                    'sentiment': content['sentiment'],
                    'polarity': content['polarity'],
                    'scraped_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                all_articles.append(article_record)
                processed_count += 1
                print(f"  Processed: {article['title'][:50]}...")
            
            print(f"Found {processed_count} new articles for '{search_term}'")
        
        # handle data retention and archiving
        output_path = 'output/news_results.csv'
        
        # load existing data
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path, parse_dates=['published_date'])
        else:
            existing_df = pd.DataFrame()
        
        # add new articles if any
        if all_articles:
            new_df = pd.DataFrame(all_articles)
            new_df['published_date'] = pd.to_datetime(new_df['published_date'])
            
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
        else:
            combined_df = existing_df.copy()
            if not combined_df.empty:
                combined_df['published_date'] = pd.to_datetime(combined_df['published_date'])
        
        # apply retention policy (3 months = 90 days)
        if not combined_df.empty:
            cutoff_date = dt.datetime.now() - dt.timedelta(days=90)
            
            # archive old data before filtering
            current_df = self.archive_old_data(combined_df, cutoff_date)
            
            # remove duplicates
            current_df = current_df.drop_duplicates(subset=['link'])
            
            # save current data
            current_df.to_csv(output_path, index=False)
            
            if all_articles:
                print(f"\nSaved {len(all_articles)} new articles")
            print(f"Current database contains {len(current_df)} articles (last 3 months)")
        else:
            print("\nNo articles found")


if __name__ == "__main__":
    scraper = NewsScaper()
    scraper.run_scraper()
