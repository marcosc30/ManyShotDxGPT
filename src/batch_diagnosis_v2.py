import os
import re
import json
import logging
from datasets import load_dataset
import requests
import pyhpo
import pandas as pd
import boto3
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)
from langchain_core.messages import HumanMessage
from tqdm import tqdm
import anthropic
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.oauth2 import service_account

from prompt_generator import PROMPT_TEMPLATE_IMPROVED
from manyshot_examples import setup_manyshot_ex
from categorize_diseases import ERN_CATEGORIES
from manyshot_examples import get_num_examples

logging.basicConfig(level=logging.INFO)

# Load the environment variables from the .env file
load_dotenv()

# models:
# -	GPT-4o
# -	GPT-4 Turbo 1106
# -	Llama 3 70B
# -	Claude 3 Opus
# -	Claude 3.5 Sonnet 



def orpha_api_get_disease_name(disease_code):
    """
    Get disease name from Orpha API
    """
    api_key = "f29dev"
    int_code = disease_code.split(":")[1]
    url = f"https://api.orphacode.org/EN/ClinicalEntity/orphacode/{int_code}/Name"
    headers = {
        "accept": "application/json",
        "apiKey": api_key
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data["Preferred term"]
    else:
        return None

def mapping_fn_with_hpo3_plus_orpha_api(data):
    """
    Same as mapping_fn but with HPO3
    This function takes in the dataset and returns the mapped dataset
    Input is a list of Example objects. Output should be another list of Example objects
    Change Phenotype object list to list of texts mapped.
    """
    pyhpo.Ontology()
    mapped_data = []
    for example in data:
        example["Phenotype"] = [pyhpo.Ontology.get_hpo_object(phenotype).name for phenotype in example["Phenotype"]]
        example["RareDisease"] = [orpha_api_get_disease_name(disease) for disease in example["RareDisease"] if disease.startswith("ORPHA:")]
        mapped_data.append(example)

    return mapped_data

# Claude 3 Opus
def initialize_anthropic_claude(prompt, temperature=0, max_tokens=2000):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    # print(message.content)
    return message

# Claude 3.5 Sonnet
def initialize_bedrock_claude(prompt, temperature=0, max_tokens=2000):
    aws_access_key_id = os.getenv("BEDROCK_USER_KEY")
    aws_secret_access_key = os.getenv("BEDROCK_USER_SECRET")
    region = "us-east-1"

    boto3_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
    )

    bedrock = boto3_session.client(service_name='bedrock-runtime')

    # body = json.dumps({
    #     "prompt": prompt,
    #     "max_tokens_to_sample": max_tokens,
    #     "top_p": 1,
    #     "temperature": temperature,
    # })

    body = json.dumps({
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "anthropic_version": "bedrock-2023-05-31"
    })

    response = bedrock.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    # claude3s = BedrockChat(
    #             client = bedrock,
    #             model_id="anthropic.claude-3-sonnet-20240229",
    #             model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens},
    # )

    print(response)

    return json.loads(response.get('body').read())

# Llama 3 70B
def initialize_azure_llama3_70b(prompt, temperature=0, max_tokens=800):
    llm = AzureMLChatOnlineEndpoint(
        endpoint_url=os.getenv("AZURE_ML_ENDPOINT_4"),
        endpoint_api_type="serverless",
        endpoint_api_key=os.getenv("AZURE_ML_API_KEY_4"),
        content_formatter=CustomOpenAIChatContentFormatter(),
        deployment_name="llama-3-70b-chat-dxgpt",
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    # logging.warning(response.content)
    return response.content

# Initialize the ChatOpenAI model turbo 1106
model_name = "gpt-4-1106-preview"
openai_api_key=os.getenv("OPENAI_API_KEY")
gpt4turbo1106 = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model_name,
        temperature = 0,
        max_tokens = 800,
    )

# gpt-4o
model_name = "gpt-4o"
openai_api_key=os.getenv("OPENAI_API_KEY")
gpt4o = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model_name,
        temperature = 0,
        max_tokens = 800,
    )

# gpt-4omini
model_name = "gpt-4o-mini"
openai_api_key=os.getenv("OPENAI_API_KEY")
gpt4omini = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model_name,
        temperature = 0,
        max_tokens = 800,
    )

def get_manyshot_diagnosis(prompt, dataset, output_file, model, include_dataset_in_ex=False):
    print("Loading data and setting up prompts")
    # Load the data
    input_path = f'data/{dataset}_categorized.csv'
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path, sep=',')
    elif input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Unsupported file extension. Please provide a .csv or .xlsx file.")
    
    # For this test, we will only use first 15 entries in the dataframe
    df = df[:15]
        
    # Create a new DataFrame to store the diagnoses
    diagnoses_df = pd.DataFrame(columns=['GT', 'Diagnosis 1', 'ERN Category'])

    # Define the chat prompt template
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    # In order to format the examples as well, we will have a dictionary of examples for each category based on this data
    ern_examples = {}
    ern_indices = []
    all_datasets = True
    print("Creating Many-Shot Examples")
    for ern_category in ERN_CATEGORIES:
        # Note: This function will print when not enough examples are present so that this can be considered, since the model
        # will not be given any inputs for the category if all are used for examples
        # To fix this, simply change the example number in the function call within a special case for the category
        num_examples = get_num_examples(ern_category, dataset)
        print(f"Using {num_examples} examples for ERN Category {ern_category}")
        curr_examples, curr_indices = setup_manyshot_ex(dataset, ern_category, example_num=num_examples, all_datasets=all_datasets, include_dataset=include_dataset_in_ex, seen_indices=ern_indices)
        ern_examples[ern_category] = curr_examples
        ern_indices += curr_indices

    if include_dataset_in_ex:
        # if this is off, you don't have to worry about dataset entries being used as a manyshot example
        if all_datasets:
            aggregate_df = pd.read_csv('data/aggregated_categorized.csv')
            first_index = aggregate_df[aggregate_df['Dataset'] == dataset].index[0]
        else:
            first_index = 0


    print("Generating Diagnoses")
    # Iterate over the rows in the synthetic data
    for index, row in tqdm(df[:200].iterrows(), total=df[:200].shape[0]):
        print(f"Entry: {index}")
        # Get the ground truth (GT) and the description
        description = row["Phenotype"]
        gt = row["RareDisease"]
        # else:
        #     gt = row[0]
        #     description = row[1]
        #     ern_category = row[2]
        # This if case checks if it is used as an example for the model, if so, it will skip it
        if include_dataset_in_ex:
            if index+first_index in ern_indices:
                continue
        # Generate a diagnosis
        diagnoses = []
        # Generate the diagnosis using the GPT-4 model
        formatted_prompt = chat_prompt.format_messages(description=description, examples=ern_examples[ern_category])
        # print(formatted_prompt[0].content)
        attempts = 0
        while attempts < 2:
            try:
                if model == "c3opus":
                    diagnosis = initialize_anthropic_claude(formatted_prompt[0].content).content[0].text
                elif model == "c3sonnet":
                    diagnosis = initialize_bedrock_claude(formatted_prompt[0].content).get("content")[0].get("text")
                    # print(diagnosis)
                elif model == "llama3_70b":
                    diagnosis = initialize_azure_llama3_70b(formatted_prompt[0].content)
                elif model == "gpt4turbo1106":
                    diagnosis = gpt4turbo1106(formatted_prompt).content
                elif model == "gpt4o":
                    diagnosis = gpt4o(formatted_prompt).content
                elif model == "gpt4omini":
                    diagnosis = gpt4omini(formatted_prompt).content
                else:
                    diagnosis = model(formatted_prompt).content  # Call the model instance directly
                break
            except Exception as e:
                attempts += 1
                print(e)
                if attempts == 2:
                    diagnosis = "ERROR"
        
        # Extract the content within the <top5> tags using regular expressions
        # print(diagnosis)
        match = re.search(r"<top5>(.*?)</top5>", diagnosis, re.DOTALL)
        # print(match)
        if match:
            diagnosis = match.group(1).strip()
        else:
            print("ERROR: <top5> tags not found in the response.")

        diagnoses.append(diagnosis)
        # print(diagnosis)

        # Add the diagnoses to the new DataFrame
        diagnoses_df.loc[index] = [gt] + [ern_category] + diagnoses

        # print(diagnoses_df.loc[index])
        # break

    # Save the diagnoses to a new CSV file
    output_path = f'data/{output_file}'
    diagnoses_df.to_csv(output_path, index=False)


# datasets = ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"]
data = load_dataset('chenxz/RareBench', "RAMEDIS", split='test')

mapped_data = mapping_fn_with_hpo3_plus_orpha_api(data)
print(type(mapped_data))

# I need to add a run that does not keep in mind categories and uses a lot more examples, I will apply that to each dataset and see results

# print(mapped_data[:5])

# Naming Guide: diagnoses_<dataset>_<model>_<shot>_<i/ni>.csv
# diagnoses: Indicator for file containing the raw diagnoses
# dataset: The dataset used
# model: The model used
# shot: Whether it uses many-shot examples or not
# i/ni: Whether dataset entries can be used as examples or not

get_manyshot_diagnosis(PROMPT_TEMPLATE_IMPROVED, 'PUMCH_ADM', 'diagnoses_PUMCH_ADM_gpt4o_manyshot_ni.csv', "gpt4o")
