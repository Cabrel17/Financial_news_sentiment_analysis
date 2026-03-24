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

    return df