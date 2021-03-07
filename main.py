import warnings

import nltk
import pandas as pd
import tweepy
from nltk.corpus import stopwords
from tweepy import OAuthHandler

warnings.filterwarnings("ignore", category=DeprecationWarning)

# downloading stopwords corpus
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('conll2000')
nltk.download('brown')
stopwords = set(stopwords.words("english"))

consumer_key = 'khYitpDgJonkCE7IMgFDGIerQ'
consumer_secret = '1gfrS6A2nk5pgwrArWCHh6DjgnRaJ0bVUBwk30djObVSJMzhlI'
access_token = '2594292031-cjboHw22k9bbT7EoXLEbFWnQOIvpmhXpGT2j3xh'
access_token_secret = 'vUJcFMZKAEEx4JJEfJqQYjNfWiYc2PrMy3p6pYzozkOCF'


# Classes
class TwitterClient(object):
    def __init__(self):
        try:
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            # add hyper parameter 'proxy' if executing from behind proxy "proxy='http://172.22.218.218:8085'"
            self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        except tweepy.TweepError as e:
            print("Tweepy Authentication Failed - \n{str(e)}")

    def get_tweets(self, query, max_tweets=1000):
        # Function to fetch tweets.
        # empty list to store parsed tweets
        tweets = []
        since_id = None
        max_id = -1
        tweet_count = 0
        tweets_per_query = 100

        while tweet_count < max_tweets:
            try:
                if max_id <= 0:
                    if not since_id:
                        new_tweets = self.api.search(q=query, count=tweets_per_query)
                    else:
                        new_tweets = self.api.search(q=query, count=tweets_per_query,
                                                     since_id=since_id)
                else:
                    if not since_id:
                        new_tweets = self.api.search(q=query, count=tweets_per_query,
                                                     max_id=str(max_id - 1))
                    else:
                        new_tweets = self.api.search(q=query, count=tweets_per_query,
                                                     max_id=str(max_id - 1),
                                                     since_id=since_id)
                if not new_tweets:
                    print("No more tweets found")
                    break

                for tweet in new_tweets:
                    parsed_tweet = {'tweets': tweet.text}

                    # appending parsed tweet to tweets list
                    if tweet.retweet_count > 0:
                        # if tweet has retweets, ensure that it is appended only once
                        if parsed_tweet not in tweets:
                            tweets.append(parsed_tweet)
                    else:
                        tweets.append(parsed_tweet)

                tweet_count += len(new_tweets)
                print("Downloaded {0} tweets".format(tweet_count))
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:
                # Just exit if any error
                print("Tweepy error : " + str(e))
                break

        return pd.DataFrame(tweets)
