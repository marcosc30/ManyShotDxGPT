import pandas as pd
# This file is to count the number of cases for each category to determine which can be analyzed
# We will create a dataframe CSV containing the total number of cases in each category, as well as the number of cases for each category per dataset
# This will also be used for further analysis in the case that the secondary category approach is used

# The data is of the shape Phenotype,RareDisease,Department,ERN Category
from categorize_diseases import ERN_CATEGORIES

def create_aggregated_csv():
    df_RAMEDIS = pd.read_csv('data/RAMEDIS_categorized.csv')
    df_RAMEDIS['Dataset'] = 'RAMEDIS'
    for index, row in df_RAMEDIS.iterrows():
        row['Dataset'] = 'RAMEDIS'
    df_HMS = pd.read_csv('data/HMS_categorized.csv')
    df_HMS['Dataset'] = 'HMS'
    for index, row in df_HMS.iterrows():
        row['Dataset'] = 'HMS'
    df_MME = pd.read_csv('data/MME_categorized.csv')
    df_MME['Dataset'] = 'MME'
    for index, row in df_MME.iterrows():
        row['Dataset'] = 'MME'
    df_LIRICAL = pd.read_csv('data/LIRICAL_categorized.csv')
    df_LIRICAL['Dataset'] = 'LIRICAL'
    for index, row in df_LIRICAL.iterrows():
        row['Dataset'] = 'LIRICAL'
    df_PUMCH_ADM = pd.read_csv('data/PUMCH_ADM_categorized.csv')
    df_PUMCH_ADM['Dataset'] = 'PUMCH_ADM'
    for index, row in df_PUMCH_ADM.iterrows():
        row['Dataset'] = 'PUMCH_ADM'
    
    df = pd.concat([df_RAMEDIS, df_HMS, df_MME, df_LIRICAL, df_PUMCH_ADM])
    df.to_csv('data/aggregated_categorized.csv', index=False)

def get_total_cases(df, ern_category):
    cases = 0
    for index, row in df.iterrows():
        if row['ERN Category'] == ern_category:
            cases += 1
    return cases

def get_cases_per_dataset(df, ern_category):
    cases = {
        'RAMEDIS': 0,
        'LIRICAL': 0,
        'PUMCH_ADM': 0,
        'MME': 0,
        'HHS': 0
    }
    for index, row in df.iterrows():
        if row['ERN Category'] == ern_category:
            if row['Dataset'] == 'RAMEDIS':
                cases['RAMEDIS'] += 1
            elif row['Dataset'] == 'LIRICAL':
                cases['LIRICAL'] += 1
            elif row['Dataset'] == 'PUMCH_ADM':
                cases['PUMCH_ADM'] += 1
            elif row['Dataset'] == 'MME':
                cases['MME'] += 1
            elif row['Dataset'] == 'HHS':
                cases['HHS'] += 1
    return cases

def make_category_info_csv():
    df = pd.read_csv('data/aggregated_categorized.csv')
    new_df = pd.DataFrame(columns=['ERN Category', 'Total Cases', 'RAMEDIS', 'LIRICAL', 'PUMCH_ADM', 'MME', 'HHS'])
    for ern_category in ERN_CATEGORIES:
        total_cases = get_total_cases(df, ern_category)
        cases_per_dataset = get_cases_per_dataset(df, ern_category)
        new_df.loc[len(new_df)] = [ern_category, total_cases, cases_per_dataset['RAMEDIS'], cases_per_dataset['LIRICAL'], cases_per_dataset['PUMCH_ADM'], cases_per_dataset['MME'], cases_per_dataset['HHS']]

    new_df.to_csv('data/category_info.csv', index=False)


create_aggregated_csv()
make_category_info_csv()
