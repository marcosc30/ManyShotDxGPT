# data is formatted like Model,Dataset,Shot,Categorized,Strict Accuracy,Lenient Accuracy
import pandas as pd

def total_accuracy(many_shot, cat):
    result_df = pd.read_csv('data/scores/results.csv')
    if many_shot and cat:
        result_df = result_df[(result_df['Shot'] == "manyshot") & (result_df['Categorized'] == "cat")]
    elif many_shot and not cat:
        result_df = result_df[(result_df['Shot'] == "manyshot") & (result_df['Categorized'] == "nocat")]
    elif not many_shot and cat:
        result_df = result_df[(result_df['Shot'] == "noshot") & (result_df['Categorized'] == "cat")]
    else:
        result_df = result_df[(result_df['Shot'] == "noshot") & (result_df['Categorized'] == "nocat")]
    strict_accuracy = result_df['Strict Accuracy'].mean()
    lenient_accuracy = result_df['Lenient Accuracy'].mean()
    print(f"Strict Accuracy: {strict_accuracy}, Lenient Accuracy: {lenient_accuracy}")


print("ManyShot, Categorized")
total_accuracy(True, True)
print("ManyShot, Not Categorized")
total_accuracy(True, True)
print("NoShot, Categorized")
total_accuracy(False, True)
print("NoShot, Not Categorized")
total_accuracy(False, False)