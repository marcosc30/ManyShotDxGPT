import os
import pandas as pd

def score_stats(model, dataset, many_shot, cat, i=False):
    if many_shot:
        many_shot = "manyshot"
    else:
        many_shot = "noshot"

    if cat:
        cat = "cat"
    else:
        cat = "nocat"

    if i:
        i = "i"
    else:
        i = "ni"
    # Load the scores data
    df = pd.read_csv(f'data/scores/{model}/scores_{dataset}_{model}_{many_shot}_{cat}_{i}.csv')

    # Summarize the data
    print(df.describe())
    print(df.head())
    print(df.shape)

    # Give me the stats for P1, P5 and P0
    print(df['Score'].value_counts())
    # Give me the stats for P1, P5 and P0 for the first 50 rows
    # print(df['Score'].iloc[:50].value_counts())

    # # To calculate the overlapping errors between the two models, we will compare the 'Score' columns from both dataframes (df and df2) to identify common mispredictions.
    # # First, filter out correct predictions (P1) as we are only interested in errors (P5 and P0).
    # errors_df_p1 = df[df['Score'] != 'P1']

    # # # Now, find the intersection of GT (Ground Truth) values in both error dataframes to identify overlapping errors.
    # # overlapping_errors = pd.merge(errors_df, errors_df2, on='GT', how='inner', suffixes=('_df', '_df2'))

    # # # Display the count of overlapping errors
    # # print(f"Number of overlapping errors: {len(overlapping_errors)}")

    # # But also for !P5
    # errors_df_p5 = df[df['Score'] == 'P0']

    # # Now, find the intersection of GT (Ground Truth) values in both error dataframes to identify overlapping errors.
    # overlapping_errors = pd.merge(errors_df, errors_df2, on='GT', how='inner', suffixes=('_df', '_df2'))

    # # Display the count of overlapping errors
    # print(f"Number of overlapping errors: {len(overlapping_errors)}")

    #Score
    count_p1 = 0
    count_p5 = 0
    count_p0 = 0

    for score in df['Score']:
        if score == 'P1':
            count_p1 += 1
        elif score == 'P2' or score == 'P3' or score == 'P4' or score == 'P5':
            count_p5 += 1
        elif score == 'P0':
            count_p0 += 1

    # print(f"Overlapping errors 14 of 23 errors in the 200 predictions: {len(overlapping_errors)/count_p0*100}%")

    # Calculate total number of predictions
    total_predictions = count_p1 + count_p5 + count_p0

    # Calculate Strict Accuracy
    strict_accuracy_new = (count_p1 / total_predictions) * 100

    # Calculate Lenient Accuracy
    lenient_accuracy_new = ((count_p1 + count_p5) / total_predictions) * 100

    print(model, dataset, strict_accuracy_new, lenient_accuracy_new)
    result_df = pd.read_csv('data/scores/results.csv')
    result_df = pd.concat([result_df, pd.DataFrame([[model, dataset, many_shot, cat, strict_accuracy_new, lenient_accuracy_new]], columns=['Model', 'Dataset', 'Shot', 'Categorized', 'Strict Accuracy', 'Lenient Accuracy'])])
    result_df.to_csv('data/scores/results.csv', index=False)


def score_stats_all_datasets(model):
    datasets = ['PUMCH_ADM', 'LIRICAL', 'HMS', 'MME', 'RAMEDIS']
    for dataset in datasets:    
        score_stats(model, dataset, True, True, False)
        score_stats(model, dataset, False, True, False)
        if dataset == 'RAMEDIS' or dataset == 'PUMCH_ADM':
            score_stats(model, dataset, True, False, False)
            score_stats(model, dataset, False, False, False)


score_stats_all_datasets('gpt4o')
score_stats_all_datasets('gpt4omini')
score_stats_all_datasets('c3opus')
score_stats_all_datasets('c3sonnet')
