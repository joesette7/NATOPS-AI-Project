import os
import numpy as np
import pandas as pd
from scipy.io import arff

DATA_DIR = "../phase1-data/NATOPS"
OUTPUT_CSV = "../results/output.csv"

def load_arff_data(path):
    data, _ = arff.loadarff(path)
    return pd.DataFrame(data)

def process_all_features(split):
    """
    For either 'train' or 'test', loads and stacks 24 dimensions vertically.
    Returns: dataframe with shape (samples * timesteps, 24), sid list, class list
    """
    feature_dfs = []
    sids = []
    labels = []

    for i in range(1, 25):
        fname = f"NATOPSDimension{i}_{split.upper()}.arff"
        path = os.path.join(DATA_DIR, fname)
        df = load_arff_data(path)

        # Extract time series and labels
        X = df.iloc[:, :-1].astype(float)  # shape: (180, 51)
        y = df.iloc[:, -1].apply(lambda x: int(float(x.decode())) if isinstance(x, bytes) else int(float(x)))  # same label per row

        # Reshape: one row per time step (samples × time steps, 1)
        X_reshaped = X.to_numpy().reshape(-1, 1)  # (180*51, 1)

        feature_dfs.append(pd.DataFrame(X_reshaped, columns=[f"fea{i}"]))

        if i == 1:
            # Build sid and label once (repeated per time step)
            n_samples, n_timesteps = X.shape
            sids = np.repeat(np.arange(n_samples), n_timesteps)
            labels = np.repeat(y.to_numpy(), n_timesteps)

    # Merge all 24 features side-by-side
    full_features = pd.concat(feature_dfs, axis=1)
    full_features['sid'] = sids
    full_features['class'] = labels
    full_features['isTest'] = 1 if split == 'test' else 0

    return full_features

def main():
    df_train = process_all_features('train')
    df_test = process_all_features('test')
    final_df = pd.concat([df_train, df_test], ignore_index=True)

    # Save output
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Output written to {OUTPUT_CSV}")
    print(f"🔍 Final shape: {final_df.shape}")
    print(final_df.head())
    
if __name__ == "__main__":
    main()