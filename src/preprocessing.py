### Libraries
import pandas as pd
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

### Variables
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

### Functions
def wrangle(path):
    """
    This function takes in argument a path and returns a dataframe.
    """
    df = pd.read_csv(path)
    df.drop(columns=df.columns[0], inplace=True) # remove an empty column
    df.drop_duplicates(inplace=True) # remove duplicates
    df[df.columns[3]] = pd.to_datetime(df[df.columns[3]], format='ISO8601', utc=True) # conversion in timestamp
    df.set_index(df.columns[3], inplace=True) # set timestamp in index

    return df

####################

def remove_stop_words(text):
    """
    This function return a document without stop words and keep lemme of the words.
    """
    text = text.lower().strip()
    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

####################

def get_polarity(text):
    """
    This function return the polarity of a text.
    """
    return TextBlob(text).sentiment.polarity

###################

def perform_lda(df, text_column, num_topics=5, num_words=5):
    """
    Applies LDA model for subject analysis on a column of cleaned text

    """
    from gensim import corpora
    from gensim.models import LdaModel
    from pandarallel import pandarallel

    pandarallel.initialize(nb_workers=4)

    headline_tokenized = df[text_column].parallel_apply(prepare_lda_data)

    dictionary = corpora.Dictionary(headline_tokenized)
    corpus = [dictionary.doc2bow(text) for text in headline_tokenized]

    # Training of LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, 
                         update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    
    # Display extract subjects
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)

    return lda_model