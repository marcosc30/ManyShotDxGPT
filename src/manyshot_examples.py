from categorize_diseases import ern_categories
import pandas as pd

# For each ern_category, we will go through the data and use the first fifteen example diseases as examples
# We will note their indices so they are not used in the testing

def get_example_diseases(data: pd.DataFrame, ern_category, example_num, seen_indices):
    # The seen_indices parameter is an optimization for when the function is used in a loop to go through all categories
    example_diseases = []
    indices = []
    for i in range(len(data)):
        if data.iloc([i][2]) == ern_category:
            example_diseases.append([data[i]['Phenotype'], data[i]['RareDisease']])
            indices.append(i)
            if len(example_diseases) == example_num:
                break
    examples_found = len(example_diseases)
    if examples_found < 15:
        print(f"Warning: ERN Category {ern_category} has only {examples_found} examples, which is less than {example_num} examples")
    return example_diseases, indices

# The prompt produced by this function is intended to be combined with the main prompt to provide many-shot examples
def setup_manyshot_ex(dataframe, ern_category, example_num=15, seen_indices=[], all_datasets=False, shuffle=True):
    if all_datasets:
        input_path = 'data/aggregated_categorized.csv'
    else: 
        input_path = f'data/{dataframe}'
    df = pd.read_csv(input_path, sep=',')
    if shuffle:
        df['Original Index'] = df.index
        df = df.sample(frac=1).reset_index(drop=True)
    example_diseases, indices = get_example_diseases(df, ern_category, example_num)
    example_disease_str = ""
    for i in range(len(example_diseases)):
        phenotype_list = ""
        for j in range(len(example_diseases[i][0])):
            phenotype_list += f"{example_diseases[i][0][j]}, "
        example_disease_str += f"{i+1}. {phenotype_list} -> {example_diseases[i][1]}\n"
    prompt = f"The following are examples of diagnosis for the ERN category '{ern_category}':\n{example_diseases}\n"
    return prompt, indices

# These are examples of how to do it without description, we could also expand it so examples also explain the reasons behind diagnosis and a T5 list

# add an optimization protocol for choosing examples like DSPy

def get_num_examples(ern_category, dataset, all_datasets=False, max_num=15, split=0.5):
    # This function is to calculate how many many-shot examples to use in case the category is too small
    # category_info.csv is formatted like ERN Category,Total Cases,RAMEDIS,LIRICAL,PUMCH ADM,MME,HHS
    input_path = 'data/category_info.csv'
    df = pd.read_csv(input_path, sep=',')
    total_cases = 0
    if all_datasets:
        for index, row in df.iterrows():
            if row['ERN Category'] == ern_category:
                total_cases += row['Total Cases']
    else:
        for index, row in df.iterrows():
            if row['ERN Category'] == ern_category:
                total_cases += row[dataset]

    if total_cases < max_num:
        return round(total_cases * split)
    else:
        return max_num