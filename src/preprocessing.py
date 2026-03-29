def wrangle(path):
    """
    This function takes in argument a path and returns a dataframe.
    """
    import pandas as pd
    import datetime
    df = pd.read_csv(path)
    df.drop(columns=df.columns[0], inplace=True) # remove an empty column
    df.drop_duplicates(inplace=True) # remove duplicates
    df[df.columns[3]] = pd.to_datetime(df[df.columns[3]], format='ISO8601', utc=True) # conversion in timestamp
    df.set_index(df.columns[3], inplace=True) # set timestamp in index

    return df

def remove_stop_words(text):
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