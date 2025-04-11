import os
import numpy as np
import pandas as pd
from scipy.io import arff

DATA_DIR = "../phase1-data/NATOPS"

OUTPUT_CSV = "../results/output.csv"

def load_arff_data(path):
    try:
        # Load ARFF data using scipy's arff reader
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        return df
    except Exception as e:
        print(f"❌ Error loading ARFF file {path}: {e}")
        return pd.DataFrame()


def flatten_dataset(df, dataset_name, dataset_type):
    n_samples, n_columns = df.shape # Get the number of samples (rows) and features (columns)

    # Check if there are any valid features (i.e., columns)
    if n_columns == 0 or n_samples == 0:
        print(f"⚠️ Dataset {dataset_name} is empty. Skipping.")
        return pd.DataFrame() # Return an empty dataframe to skip this dataset

    # Check if 'classAttribute' column exists (i.e., whether the dataset contains labels)
    if 'classAttribute' in df.columns:
        # Extract labels (class column) and decode bytes to strings if necessary
        y = df['classAttribute'].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        df = df.drop(columns=['classAttribute']) # Drop the 'classAttribute' column from the dataframe
    else:
        y = [None] * n_samples # If no class column, assign None to labels

    # Check if the dataframe is still empty after cleaning
    if df.empty:
        print(f"⚠️ Dataset {dataset_name} has no valid features. Skipping.")
        return pd.DataFrame() # Return an empty dataframe to skip this dataset

    time_steps = df.shape[1] # Each column represents one time step in the dataset
    sample_ids = np.repeat(np.arange(n_samples), time_steps) # Repeating sample IDs for each time step
    time_steps_repeat = np.tile(np.arange(time_steps), n_samples) # Repeating time step indices for each sample
    flat_data = df.to_numpy().reshape(-1, 1) # Flatten the dataset into a 1-dimensional array

    # Creating a flattened DataFrame with new structure
    flattened_df = pd.DataFrame({
    'isTest': np.repeat(1 if dataset_type == 'test' else 0, n_samples * time_steps), # Label whether it's a test dataset
    'dataset': dataset_name, # Name of the dataset (e.g., 'dataset1')
    'dataset_type': dataset_type, # Type of dataset (train or test)
    'sid': sample_ids, # Sample ID (sid) repeated for each time step
    'time_step': time_steps_repeat, # Time step for each data point
    'value': flat_data.flatten(), # Flattened values of the original dataset
    'label': np.repeat(y, time_steps) # Repeating labels for each time step
    })

    # Generate feature columns dynamically excluding 'classAttribute'
    feature_columns = [f'channel_0_{i}' for i in range(n_columns - 1)] # Create feature names dynamically
    flattened_df[feature_columns] = pd.DataFrame(df.values, columns=feature_columns) # Assign the feature columns

    # Reorder the columns to match the desired output
    flattened_df = flattened_df[['isTest'] + feature_columns + ['sid', 'label']] # Reorder columns

    return flattened_df # Return the flattened dataframe

def main():
    all_dfs = [] # Initialize a list to store the processed dataframes
    print(f"🔍 Looking in folder: {DATA_DIR}") # Print the directory being searched
    files = os.listdir(DATA_DIR) # List all files in the DATA_DIR folder

    print(f"Files in directory: {files}") # Log the files found in the directory

    # Loop through all files in the directory
    for filename in files:
        # Skip files that do not end with "_TRAIN.arff"
        if not filename.endswith("_TRAIN.arff"):
            continue

        base_name = filename.replace("_TRAIN.arff", "") # Extract the base name (remove '_TRAIN.arff')
        train_path = os.path.join(DATA_DIR, f"{base_name}_TRAIN.arff") # Path to the training file
        test_path = os.path.join(DATA_DIR, f"{base_name}_TEST.arff") # Path to the test file

        # Check if the corresponding test file exists
        if not os.path.exists(test_path):
            print(f"⚠️ Skipping {base_name} — test file not found.")
            continue # Skip this dataset and move to the next file

        print(f"\n📦 Processing dataset: {base_name} (train: {train_path}, test: {test_path})")

        # Load training and test data using the load_arff_data function
        df_train = load_arff_data(train_path)
        df_test = load_arff_data(test_path)

        # If either train or test data is empty, skip this dataset
        if df_train.empty or df_test.empty:
            print(f"⚠️ Skipping {base_name} due to empty train or test data.")
            continue

        print(f"📐 Train shape: {df_train.shape}, Test shape: {df_test.shape}")

        # Flatten both training and test datasets
        df_train_flat = flatten_dataset(df_train, dataset_name=base_name, dataset_type="train")
        df_test_flat = flatten_dataset(df_test, dataset_name=base_name, dataset_type="test")

        # Append the flattened dataframes
        if not df_train_flat.empty and not df_test_flat.empty:
            all_dfs.append(pd.concat([df_train_flat, df_test_flat], ignore_index=True))
        else:
            print(f"⚠️ Skipping {base_name} due to empty flattened data.")

    # Check if any datasets were processed
    if not all_dfs:
        print("❌ No datasets were processed.")
        return # Exit the function if no data was processed

    # Concatenate all flattened dataframes into a single final dataframe
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Save the final dataframe to a CSV file
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Done! Flattened data saved to: {OUTPUT_CSV}") # Confirmation message
    print(f"📏 Final shape: {final_df.shape}") # Print the shape of the final flattened dataframe

# Entry point for the script
if __name__ == "__main__":
    main() # Call the main function to run the script