import pandas as pd

# Reading csvs
df1 = pd.read_csv('dataset/metadata.csv')
df2 = pd.read_csv('results.csv')

# Removing 'category' column from the metadata file
df1 = df1.drop('category', axis=1)

# Merge of the two dataframes maintaining the first dataframe structure
df_merged = df1.merge(df2, left_on='img_path', right_on='image_path', how='left')
df_merged = df_merged.drop('image_path', axis=1)

df_merged = df_merged.reset_index(drop=True)

df_merged.to_csv('metadata_classified.csv', index=False)

print("File saved as 'metadata_classified.csv'")
print("\nFirst 5 rows of the result:")
print(df_merged.head())

# Verifica se ci sono path non matchati
unmatched = df_merged[df_merged['gender'].isna()]
if not unmatched.empty:
    print(f"\nWarning: {len(unmatched)} rows didn't find a match:")
    print(unmatched[['filename', 'img_path']])  # Corretto da 'path' a 'img_path'