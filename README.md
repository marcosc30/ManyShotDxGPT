# Accuracy of LLMs for Rare Disease Diagnostics Across Categories with Many-Shot Learning

### Intro to this repo

Welcome to our repository dedicated to the evaluation of [DxGPT](https://dxgpt.app/) across various AI models with many-shot learning, across different categories of rare diseases. 

This project is, in a way, a continuation of a previous paper (https://www.medrxiv.org/content/10.1101/2024.05.08.24307062v1) evaluating the straight effectiveness of different LLMs in diagnosing rare diseases.

This project looks to investigate two further main questions. 1. Which types of rare diseases do LLM models excel at diagnosing? Do LLMs perform better given diseases that affect certain biological systems? 2. Given the success that many-shot learning has had at improving LLM results in certain tasks, how much, if any, improvement is seen when applying many-shot learning to LLM diagnosis?

This repository mainly contains the code created to carry out the tests, as well as the data gathered in the tests. The data is quite extensive, so we recommend becoming familiar with the naming conventions to navigate through it more easily. We may also add notebooks and plots as the analysis of the data continues.

### Summary plots



### IPynb Dashboard with comparison of multiple models:


## File naming convention

The naming convention of the files in this repository is systematic and provides quick insights into the contents and purpose of each file. Understanding the naming structure will help you navigate and utilize the data effectively.

### Structure

Each file name is composed of four main parts:

1. **Evaluation data prefix**: All files related to model evaluation scores begin with `scores_`. This prefix is a clear indicator that the file contains data from the evaluation 
process. `diagnoses_` prefix is used for the files that contain the actual diagnoses from each test run, same naming convention as the scores files.

2. **Dataset**: The dataset name is included to provide context. Example datasets include:
    - `RAMEDIS` is the RAMEDIS dataset from RareBench 
    - `PUMCH_ADM` is the PUMCH dataset from RareBench
    - `MME` is the MME dataset from RareBench
    - `HHS` is the HHS dataset from RareBench
    - `aggregated` is all of the datasets from Rarebench aggregated into one

3. **Model identifier**: Following the prefix, the name includes an identifier for the AI model used during the evaluation. Some of the possible model identifiers are:
   - `gpt4o`: Data evaluated using the GPT-4o Model
   - `_llama3_70b`: Data evaluated using the LLaMA 3 70b model.
   - `_c3opus`: Data evaluated using the Claude 3 Opus model.
4. **Shot**: Identifies whether it was a many-shot or no-shot test
5. **Categorized** Identifies if categories were taken into account when generating examples or if examples were general (cat/nocat)
6. **Dataset Examples Included**: Identifies whether or not examples were taken from the dataset being tested (i/ni). All of the tests so far have been run ni, but the option is there in case further testing is done.

This structured approach to file naming ensures that each file is easily identifiable and that its contents are self-explanatory based on the name alone.

## Evaluation metrics

- **Strict Accuracy (P1)**: Top suggestion matches the ground truth.
- **Top-5 Accuracy (P1+P5)**: Ground truth appears within the top 5 suggestions.


