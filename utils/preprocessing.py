import pandas as pd

def encode_drugs(df):
    all_drugs = list(set(df[['drug1','drug2','drug3','drug4']].values.flatten()))

    def encode(row):
        features = {drug: 0 for drug in all_drugs}
        for d in ['drug1','drug2','drug3','drug4']:
            features[row[d]] = 1
        return pd.Series(features)

    return df.apply(encode, axis=1), all_drugs