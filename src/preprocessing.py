def preprocess1(path):
    """ 
    This function takes in argument a path and returns a dataframe.
    """
    import pandas as pd
    import datetime
    df = pd.read_csv(path)
    df.drop_duplicates(subset=df.columns[2], inplace=True) # drop duplicates from url column
    df.drop(columns=df.columns[0], inplace=True) # remove an useless column, here 'Unnamed 0'
    df[df.columns[3]] = pd.to_datetime(df[df.columns[3]], format='ISO8601', utc=True) # conversion in timestamp
    df.set_index(df.columns[3], inplace=True) # set timestamp in index

    return df


def preprocess2(path):
    """
    This function takes in argument a path and returns a dataframe
    """
    import pandas as pd
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.drop_duplicates(subset=df.columns[0], inplace=True) # removing of duplicates from headline

    return df


def process_text1(text):
    """
    This function return a document without stop words and keep lemme of the words.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer 
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    # Transforming in lower characters and removing of empty spaces
    text = text.lower().strip()

    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Drop stop_words
    stop_words = set(stopwords.words('english'))
    cleaned_tokens_list = [w for w in tokens if w not in stop_words]

    #stemming 
    stem = PorterStemmer()
    tokens = [stem.stem(w) for w in cleaned_tokens_list]

     # cleaned_text 
    cleaned_text = " ".join(tokens)

    return cleaned_text


def get_polarity(text):
    from textblob import TextBlob
    return TextBlob(text).sentiment.polarity


def display_tokens_info(tokens):
    """display info about corpus"""
    print(f"nb tokens {len(tokens)}, nb tokens unique {len(set(tokens))}")


def prepare_lda_data(text):
    """
    Prepares the text for LDA by tokenizing it.
    """
    # Tokenization by space
    return text.split()


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
    
