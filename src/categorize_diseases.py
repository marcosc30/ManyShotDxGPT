# Script to categorize the diseases in rarebench into the ERN categories using GPT

from datasets import load_dataset
import requests
import pyhpo
import boto3
import json
import os
import pandas as pd
#from batch_diagnosis_v2 import mapping_fn_with_hpo3_plus_orpha_api

ern_categories = ['Rare endocrine conditions', 'Rare kidney diseases', 'Rare bone disorders', 'Rare and complex epilepsies',
                        'Rare adult solid cancers', 'Rare urogenital diseases and complex conditions', 'Neuromuscular diseases',
                        'Rare genetic tumour risk syndromes', 'Uncommon and rare diseases of the heart', 'Paediatric cancer (haemato-oncology)',
                        'Rare hepatological diseases', 'Rare connective tissue and musculoskeletal diseases', 'Rare immunodeficiency', 'Autoinflammatory and autoimmune diseases'
                        'Transplantation in children', 'Rare haematological diseases', 'Rare eye diseases', 'Rare malformation syndromes, intellectual and other neurodevelopmental disorders', 
                        'Rare respiratory disease', 'Rare neurological diseases', 'Rare, complex, and undiagnosed skin disorders',  'Rare inherited and congenital (digestive and gastrointestinal) anomalies', 
                        'Hereditary metabolic disorders', 'Rare multisystemic vascular diseases']

# The ERNs currently cover the main clusters of rare, complex, and low-prevalence diseases.
# Endo-ERN	European Reference Network on rare endocrine conditions
# ERKNet	European Reference Network on rare kidney diseases
# ERN BOND	European Reference Network on rare bone disorders
# ERN CRANIO	European Reference Network on rare craniofacial anomalies and ear, nose and throat (ENT) disorders
# ERN EpiCARE	European Reference Network on rare and complex epilepsies
# ERN EURACAN	European Reference Network on rare adult solid cancers
# ERN eUROGEN	European Reference Network on rare urogenital diseases and complex conditions
# ERN EURO-NMD	European Reference Network on neuromuscular diseases
# ERN GENTURIS	European Reference Network on rare genetic tumour risk syndromes
# ERN GUARD-HEART	European Reference Network on uncommon and rare diseases of the heart
# ERN PaedCan	European Reference Network on paediatric cancer (haemato-oncology)
# ERN RARE-LIVER	European Reference Network on rare hepatological diseases
# ERN ReCONNET	European Reference Network on rare connective tissue and musculoskeletal diseases
# ERN RITA	European Reference Network on rare immunodeficiency, autoinflammatory and autoimmune diseases
# ERN TRANSPLANT-CHILD	European Reference Network on transplantation in children
# ERN-EuroBloodNet	European Reference Network on rare haematological diseases
# ERN-EYE	European Reference Network on rare eye diseases
# ERN-ITHACA	European Reference Network on rare malformation syndromes, intellectual and other neurodevelopmental disorders
# ERN-LUNG	European Reference Network on rare respiratory diseases
# ERN-RND	European Reference Network on rare neurological diseases
# ERN-Skin	European Reference Network on rare, complex, and undiagnosed skin disorders
# ERNICA	European Reference Network on rare inherited and congenital (digestive and gastrointestinal) anomalies
# MetabERN	European Reference Network on hereditary metabolic disorders
# VASCERN	European Reference Network on rare multisystemic vascular diseases

# import these next two functions from batch_diagnosis_v2 once the API keys work and the file can run
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


datasets = ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"]

# we initialize the claude 3.5 sonnet model here
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

    #print(response)

    return json.loads(response.get('body').read())


def categorize_diseases(dataset):
        print(f"Loading dataset {dataset}")
        data = load_dataset('chenxz/RareBench', dataset, split='test', trust_remote_code=True)
        # let's try with first 10 first
        print("Mapping data to disease names")
        data = mapping_fn_with_hpo3_plus_orpha_api(data)
        print("Data mapped")
        data = data
        # Format of the data is {Phenotype: [], RareDisease: []}
        # Categorizing diseases into ERN categories
        for i in range(len(data)):
            print("Entry ", i)
            # Get the disease from the data
            disease = data[i]['RareDisease']
            phenotype = data[i]['Phenotype']
            # Generate the prompt for the GPT model
            prompt = f"Categorize the disease '{disease}' into an ERN category from this list ({ern_categories}) using known information about the disease. Please only provide just the category followed by '\n' so it can be split easily:" 

            # Generate the completion using the GPT model
            response = initialize_bedrock_claude(prompt).get("content")[0].get("text")
            # print(disease)
            # print(response)

            # Check if response is a string before splitting
            if isinstance(response, str):
                category = response.split('\n')[0]
            else:
                print(f"Response is not a string at index {i}: {response}")
                continue

            # Get the generated category from the Claude response
            if category not in ern_categories:
                response = initialize_bedrock_claude(prompt)
                if isinstance(response, str):
                    category = response.split('\n')[0]
                else:
                    print(f"Response is not a string at index {i}: {response}")
                    continue

                if category not in ern_categories:
                    print(f"Category '{category}' not recognized, index {i}")
                    continue


            # Add the category field to the data entry
            data[i]['ERN Category'] = category
        return data

# Save the recategorized data
def dataset_categorization(dataset):
    data = categorize_diseases(dataset)
    data_df = pd.DataFrame(data)
    output_path = f'data/{dataset}_categorized.csv'
    data_df.to_csv(output_path, index=False)

# RAMEDIS_categorize() # RAMEDIS is all metabolic diseases, so no need to categorize
# dataset_categorization("MME")
# dataset_categorization("HMS")
# dataset_categorization("LIRICAL")
# dataset_categorization("PUMCH_ADM")

def RAMEDIS_categorize():
    data = load_dataset('chenxz/RareBench', 'RAMEDIS', split='test', trust_remote_code=True)
    data = mapping_fn_with_hpo3_plus_orpha_api(data)
    for example in data:
        example["ERN Category"] = "Hereditary metabolic disorders"
    data_df = pd.DataFrame(data)
    output_path = 'data/RAMEDIS_recategorized.csv'
    data_df.to_csv(output_path, index=False)
    return data