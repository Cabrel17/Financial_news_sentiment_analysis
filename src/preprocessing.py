### Libraries
import pandas as pd
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from gensim import corpora
from gensim.models import Phrases, LdaModel

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
    # Creation of bigrams
    raw_tokens = df[text_column].apply(lambda x: str(x).split()).tolist()
    bigram_transformer = Phrases(raw_tokens, min_count=5, threshold=10)
    texts = [bigram_transformer[doc] for doc in raw_tokens]

    # Creation of the dictionary and corpus for LDA
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Training of the LDA model
    lda_model = LdaModel(
        corpus=corpus, 
        id2word=dictionary, 
        num_topics=num_topics, 
        random_state=42, 
        passes=15, 
        alpha='auto'
    )

    # Function to get the main topic for each document
    def get_main_topic(bow):
        topics = lda_model.get_document_topics(bow)
        return max(topics, key=lambda x: x[1]) # Retourne (ID, Probabilité)

    # Dataframe enrichment
    topic_info = [get_main_topic(bow) for bow in corpus]
    df['Topic_ID'] = [t[0] for t in topic_info]
    df['Topic_Score'] = [t[1] for t in topic_info]
    
    # Extraction of key words
    keywords = {i: [word for word, prop in lda_model.show_topic(i, num_words)] 
                for i in range(num_topics)}
    
    df['Topic_Keywords'] = df['Topic_ID'].map(keywords)

    # Display topics
    print("--- Synthèse des Thématiques ---")
    for i, words in keywords.items():
        print(f"Thème {i}: {', '.join(words)}")

    return lda_model, df