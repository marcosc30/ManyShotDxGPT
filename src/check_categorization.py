from categorize_diseases import ERN_CATEGORIES
import pandas as pd

def check_categorization(dataset):
    # Data is formatted as Phenotype,RareDisease,Department,ERN Category
    df = pd.read_csv(f"data/{dataset}_categorized.csv")
    for index, row in df.iterrows():
        if row["ERN Category"] not in ERN_CATEGORIES:
            print(f"Error: {row['RareDisease']} is not categorized correctly in row {index}")
            return False
        
    return True

check_categorization("LIRICAL")
check_categorization("HMS")