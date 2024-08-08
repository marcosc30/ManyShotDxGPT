from categorize_diseases import ern_categories
import pandas as pd

# For each ern_category, we will go through the data and use the first fifteen example diseases as examples
# We will note their indices so they are not used in the testing

def get_example_diseases(data: pd.DataFrame, ern_category, example_num, seen_indices):
    # The seen_indices parameter is a not yet implemented optimization for when the function is used in a loop to go through all categories
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
def setup_manyshot_ex(dataset, ern_category, example_num=15, seen_indices=[], all_datasets=True, include_dataset=False, shuffle=True):
    # If all datasets is true, indices provided will be relative to the aggregative csv
    # Otherwise, they are relative to that dataset
    # In theory, they can be converted to each other by just adding or subtracting the index of the first element in the dataset in the aggregative csv
    if all_datasets:
        input_path = 'data/aggregated_categorized.csv'
    else: 
        input_path = f'data/{dataset}'
    df = pd.read_csv(input_path, sep=',')
    if shuffle:
        df['Original Index'] = df.index
        df = df.sample(frac=1).reset_index(drop=True)
    if not include_dataset:
        # we will remove all of the entries from the dataset from the df
        df = df[df['Dataset'] != dataset]

    example_diseases, indices = get_example_diseases(df, ern_category, example_num, seen_indices)
    if shuffle:
        for index in indices:
            index = df['Original Index'][index]

    example_disease_str = ""
    for i in range(len(example_diseases)):
        phenotype_list = ""
        for j in range(len(example_diseases[i][0])):
            phenotype_list += f"{example_diseases[i][0][j]}, "
        example_disease_str += f"{i+1}. {phenotype_list} -> {example_diseases[i][1]}\n"
    prompt = f"The following are examples of diagnosis for the ERN category '{ern_category}':\n{example_diseases}\n"
    return prompt, indices

def get_num_examples(ern_category, dataset, all_datasets=True, include_dataset=False, max_num=15, split=0.5):
    # This function is to calculate how many many-shot examples to use in case the category is too small
    # category_info.csv is formatted like ERN Category,Total Cases,RAMEDIS,LIRICAL,PUMCH ADM,MME,HHS
    input_path = 'data/category_info.csv'
    df = pd.read_csv(input_path, sep=',')
    total_cases = 0
    if all_datasets:
        for index, row in df.iterrows():
            if row['ERN Category'] == ern_category:
                total_cases += row['Total Cases'] 
                if not include_dataset:
                    total_cases -= row[dataset]
    else:
        if not include_dataset:
            return "Error in calculating number of examples: Please set include_dataset to True if you want to use a specific dataset"
        for index, row in df.iterrows():
            if row['ERN Category'] == ern_category:
                total_cases += row[dataset]

    if total_cases < max_num:
        return round(total_cases * split)
    else:
        return max_num
    

# These are examples of how to do it without description, we could also expand it so examples also explain the reasons behind diagnosis and a T5 list

# add an optimization protocol for choosing examples like DSPy