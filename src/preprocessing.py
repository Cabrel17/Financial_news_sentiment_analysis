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
    This function return a list of tokens without stop words.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer 

    # Transforming in lower characters and removing of empty spaces
    text = text.lower().strip()

    # Tokenization
    tokenizer = RegexpTokenizer(r'\$[\d,.]+|€[\d,.]+|£[\d,.]+|\d+\.\d+|\d+|[\w]+')
    tokens = tokenizer.tokenize(text)

    # Drop stop_words
    stop_words = set(stopwords.words('english'))
    cleaned_tokens_list = [w for w in tokens if w not in stop_words]

    return cleaned_tokens_list


def display_tokens_info(tokens):
    """display info about corpus"""
    print(f"nb tokens {len(tokens)}, nb tokens unique {len(set(tokens))}")


def process_text2(text, rare_tokens):
    """
    This function return a list of tokens without stop words and rare_tokens.
    """
    import pandas as pd

    tokens = process_text1(text) # use process_text1 to return a token list without stop words

    # keep tokens whose nb characters > 1
    tokens = [w for w in tokens if len(w) > 1]
    
    # construction of a list of tokens without rare_tokens
    tokens = [w for w in tokens if w not in rare_tokens] 

    

    return tokens
    