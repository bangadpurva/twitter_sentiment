import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tweepy
from textblob import TextBlob
from textblob.en.np_extractors import ConllExtractor
from tweepy import OAuthHandler
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=DeprecationWarning)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

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


class PhraseExtractHelper(object):
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()

    def leaves(self, tree):
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            yield subtree.leaves()

    def normalise(self, word):
        word = word.lower()
        word = self.lemmatizer.lemmatize(word)
        return word

    def acceptable_word(self, word):
        accepted = bool(3 <= len(word) <= 40
                        and word.lower() not in stopwords
                        and 'https' not in word.lower()
                        and 'http' not in word.lower()
                        and '#' not in word.lower()
                        )
        return accepted

    def get_terms(self, tree):
        for leaf in self.leaves(tree):
            term = [self.normalise(w) for w, t in leaf if self.acceptable_word(w)]
            yield term


twitter_client = TwitterClient()

tweets_df = twitter_client.get_tweets('Pandemic', max_tweets=1000)
print(f'tweets_df Shape - {tweets_df.shape}')


def fetch_sentiment_using_textblob(text):
    analysis = TextBlob(text)
    return 'pos' if analysis.sentiment.polarity >= 0 else 'neg'


sentiments_using_textblob = tweets_df.tweets.apply(lambda tweet: fetch_sentiment_using_textblob(tweet))
pd.DataFrame(sentiments_using_textblob.value_counts())
tweets_df['sentiment'] = sentiments_using_textblob


def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    return text


tweets_df['cleaned_tweets'] = np.vectorize(remove_pattern)(tweets_df['tweets'], "@[\w]*: | *RT*")

cleaned_tweets = []

for index, row in tweets_df.iterrows():
    words_without_links = [word for word in row.cleaned_tweets.split() if 'http' not in word]
    cleaned_tweets.append(' '.join(words_without_links))

tweets_df['cleaned_tweets'] = cleaned_tweets
tweets_df = tweets_df[tweets_df['cleaned_tweets'] != '']
tweets_df.drop_duplicates(subset=['cleaned_tweets'], keep=False)
tweets_df = tweets_df.reset_index(drop=True)

tweets_df['absolute_cleaned_tweets'] = tweets_df['cleaned_tweets'].str.replace("[^a-zA-Z# ]", "")
stopwords_set = set(stopwords)
cleaned_tweets = []
for index, row in tweets_df.iterrows():
    words_without_stopwords = [word for word in row.absolute_cleaned_tweets.split()
                               if not word in stopwords_set and '#' not in word.lower()]

    cleaned_tweets.append(' '.join(words_without_stopwords))
tweets_df['absolute_cleaned_tweets'] = cleaned_tweets

tokenized_tweet = tweets_df['absolute_cleaned_tweets'].apply(lambda x: x.split())
word_lemmatizer = WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
for i, tokens in enumerate(tokenized_tweet):
    tokenized_tweet[i] = ' '.join(tokens)
tweets_df['absolute_cleaned_tweets'] = tokenized_tweet

sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""
chunker = nltk.RegexpParser(grammar)

key_phrases = []
phrase_extract_helper = PhraseExtractHelper()

for index, row in tweets_df.iterrows():
    toks = nltk.regexp_tokenize(row.cleaned_tweets, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)

    terms = phrase_extract_helper.get_terms(tree)
    tweet_phrases = []

    for term in terms:
        if len(term):
            tweet_phrases.append(' '.join(term))

    key_phrases.append(tweet_phrases)

textblob_key_phrases = []
extractor = ConllExtractor()

for index, row in tweets_df.iterrows():
    words_without_hash = [word for word in row.cleaned_tweets.split() if '#' not in word.lower()]

    hash_removed_sentence = ' '.join(words_without_hash)

    blob = TextBlob(hash_removed_sentence, np_extractor=extractor)
    textblob_key_phrases.append(list(blob.noun_phrases))

tweets_df['key_phrases'] = textblob_key_phrases


def generate_word_cloud(all_words):
    word_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5,
                           colormap='Dark2').generate(all_words)

    plt.figure(figsize=(14, 10))
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


all_words = ' '.join([text for text in tweets_df['absolute_cleaned_tweets'][tweets_df.sentiment == 'pos']])
generate_word_cloud(all_words)

all_words = ' '.join([text for text in tweets_df['absolute_cleaned_tweets'][tweets_df.sentiment == 'neg']])
generate_word_cloud(all_words)


def hashtag_extract(text_list):
    hashtags = []
    for text in text_list:
        ht = re.findall(r"#(\w+)", text)
        hashtags.append(ht)
    return hashtags


def generate_hashtag_freqdist(hashtags):
    a = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'#': list(a.keys()),
                      'count': list(a.values())})
    # selecting to p 15 most frequent hashtags
    d = d.nlargest(columns="count", n=25)
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(data=d, x="#", y="count")
    plt.xticks(rotation=80)
    ax.set(ylabel='count')
    plt.show()


hashtags = hashtag_extract(tweets_df['cleaned_tweets'])
hashtags = sum(hashtags, [])
generate_hashtag_freqdist(hashtags)

bow_word_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
bow_word_feature = bow_word_vectorizer.fit_transform(tweets_df['absolute_cleaned_tweets'])
target_variable = tweets_df['sentiment'].apply(lambda x: 0 if x == 'neg' else 1)

tweets_df2 = tweets_df[tweets_df['key_phrases'].str.len() > 0]
phrase_sents = tweets_df2['key_phrases'].apply(lambda x: ' '.join(x))
bow_phrase_vectorizer = CountVectorizer(max_df=0.90, min_df=2)
bow_phrase_feature = bow_phrase_vectorizer.fit_transform(phrase_sents)

target_variable = tweets_df2['sentiment'].apply(lambda x: 0 if x == 'neg' else 1)


def plot_confusion_matrix(matrix):
    plt.clf()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Set2_r)
    class_names = ['Positive', 'Negative']
    plt.title('Confusion Matrix')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    s = [['TP', 'FP'], ['FN', 'TN']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(matrix[i][j]))
    plt.show()


def naive_model(X_train, X_test, y_train, y_test):
    naive_classifier = GaussianNB()
    naive_classifier.fit(X_train.toarray(), y_train)
    predictions = naive_classifier.predict(X_test.toarray())
    print(f'Accuracy Score - {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)


# X_train, X_test, y_train, y_test = train_test_split(bow_word_feature, target_variable, test_size=0.3, random_state=272)
# naive_model(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(bow_phrase_feature, target_variable, test_size=0.3,
                                                    random_state=272)
naive_model(X_train, X_test, y_train, y_test)
