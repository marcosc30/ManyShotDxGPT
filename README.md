# Accuracy of LLMs for Rare Disease Diagnostics Across Categories with Many-Shot Learning

### Intro to this repo

Welcome to our repository dedicated to the evaluation of [DxGPT](https://dxgpt.app/) across various AI models, for rare diseases. 

This project looks to investigate two main questions. 1. Which types of rare diseases do LLM models excel at diagnosing? Do LLMs perform better given diseases that affect certain biological systems? 2. Given the success that many-shot learning has had at improving LLM results in certain tasks, how much, if any, improvement is seen when applying many-shot learning to LLM diagnosis?

This repository, once finished, should contain the could used to run the tests, as well as detailed notebooks with comparisons and data gathered throughout the trials. 

Stay tuned for updates and findings as we delve deeper into the world of AI and healthcare.

### Summary plots



### IPynb Dashboard with comparison of multiple models:


## File naming convention

The naming convention of the files in this repository is systematic and provides quick insights into the contents and purpose of each file. Understanding the naming structure will help you navigate and utilize the data effectively.

### Structure

Each file name is composed of four main parts:

1. **Evaluation data prefix**: All files related to model evaluation scores begin with `scores_`. This prefix is a clear indicator that the file contains data from the evaluation 
process. `diagnoses_` prefix is used for the files that contain the actual diagnoses from each test run, same naming convention as the scores files. `synthetic_*` prefix is used for the synthetic datasets.

2. Additionally, the dataset name is included to provide context. Example datasets include:
    - `(empty)` is a gpt4 synthetic dataset
    - `claude` is a claude 2 synthetic dataset
    - `medisearch` is a medisearch synthetic dataset
    - `RAMEDIS` is the RAMEDIS dataset from RareBench 
    - `PUMCH_ADM` is the PUMCH dataset from RareBench
    - `URG_Torre_Dic_200` is our proprietary dataset from common diseases in urgency care.

3. This is followed by the version of the dataset used for the evaluation (`(empty)` is v1, `v2` is the second version of the dataset).

4. **Model identifier**: Following the prefix, the name includes an identifier for the AI model used during the evaluation. Some of the possible model identifiers are:
   - `_gpt4_0613`: Data evaluated using the GPT-4 model checkpoint 0613.
   - `_llama`: Data evaluated using the LLaMA model.
   - `_c3`: Data evaluated using the Claude 3 model.
   - `_mistral`: Data evaluated using the Mistral model.
   - `_geminipro15`: Data evaluated using the Gemini Pro 1.5 model.

### Modifiers

In addition to the main parts, file names may include modifiers that provide further context about the evaluation:

- `_improved`: Indicates that the file contains data from an evaluation using an improved version of the prompt.
- `_rare_only_prompt`: Specifies that the evaluation prompt was a test focused exclusively on rare diseases.

### Examples

- `scores_v2_gpt4_0613.csv`: Evaluation scores from the second version of the dataset using the GPT-4 model checkpoint 0613.
- `scores_medisearch_v2_gpt4turbo1106.csv`: Evaluation scores from the medisearch synthetic dataset using the GPT-4 model turbo checkpoint 1106.
- `scores_URG_Torre_Dic_200_improved_c3sonnet.csv`: Evaluation scores from the urgency care dataset from December using the Claude 3 Sonnet model with an improved prompt.
- `scores_RAMEDIS_cohere_cplus.csv`: Evaluation scores from the RAMEDIS dataset using the Cohere Command R + model.
- `scores_PUMCH_ADM_mistralmoe.csv`: Evaluation scores from the PUMCH dataset using the Mistral MoE 8x7B model.

This structured approach to file naming ensures that each file is easily identifiable and that its contents are self-explanatory based on the name alone.


## Evaluation metrics

- **Strict Accuracy (P1)**: Top suggestion matches the ground truth.
- **Top-5 Accuracy (P1+P5)**: Ground truth appears within the top 5 suggestions.


