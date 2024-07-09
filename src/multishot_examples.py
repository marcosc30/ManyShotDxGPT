from categorize_diseases import ern_categories

# For each ern_category, we will go through the data and use the first fifteen example diseases as examples
# We will note their indices so they are not used in the testing

def get_example_diseases(data, ern_category):
    example_diseases = []
    for i in range(len(data)):
        if data[i]['ERN Category'] == ern_category:
            example_diseases.append([data[i]['Phenotype'], data[i]['RareDisease']])
            if len(example_diseases) == 15:
                break
    if len(example_diseases) < 15:
        print(f"ERN Category {ern_category} has less than 15 examples")
    return example_diseases

# The prompt produced by this function is intended to be combined with the main prompt to provide many-shot examples
def setup_manyshot_ex(data, ern_category):
    example_diseases = get_example_diseases(data, ern_category)
    example_disease_str = ""
    for i in range(len(example_diseases)):
        phenotype_list = ""
        for j in range(len(example_diseases[i][0])):
            phenotype_list += f"{example_diseases[i][0][j]}, "
        example_disease_str += f"{i+1}. {phenotype_list} -> {example_diseases[i][1]}\n"
    prompt = f"The following are examples of diagnosis for the ERN category '{ern_category}':\n{example_diseases}\n"
    return prompt
