def preprocess1(path):
    """ 
    This function takes in argument a path and returns a dataframe.
    """
    import pandas as pd
    import datetime
    df = pd.read_csv(path)
    df.drop_duplicates(subset=df.columns[2], inplace=True)
    df.drop(columns=df.columns[0], inplace=True)
    df[df.columns[3]] = pd.to_datetime(df[df.columns[3]], format='ISO8601', utc=True)
    df.set_index(df.columns[3], inplace=True)

    return df