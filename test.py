import pandas as pd

file_path = '/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_en/data/processed/wp/test_en.csv'
df = pd.read_csv(file_path)
number_of_entries = len(df)
# Or, equivalently: number_of_entries = df.shape[0]

# print(df.head())
print(f"The file '{file_path}' has {number_of_entries} data entries (excluding the header).")
print("Column names:", df.columns.tolist())

label_counts = df['label'].value_counts()
print("Label counts:")
print(label_counts)