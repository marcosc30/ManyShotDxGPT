# Script to categorize the diseases in rarebench into the ERN categories using GPT

from datasets import load_dataset
import requests
import pyhpo
import openai
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

def categorize_diseases(dataset):
        data = load_dataset('chenxz/RareBench', dataset, split='test', trust_remote_code=True)
        data = mapping_fn_with_hpo3_plus_orpha_api(data)
        # Format of the data is {Phenotype: [], RareDisease: []}
        for i in range(len(data)):
            # Define the ERN categories
            # Initialize the OpenAI GPT model
            openai.api_key = 'API_KEY'
            model = "gpt-3.5-turbo"

            # Iterate over the data
            for i in range(len(data)):
                # Get the disease from the data
                disease = data[i]['RareDisease']
                # Generate the prompt for the GPT model
                prompt = f"Categorize the disease '{disease}' into an ERN category:"

                # Generate the completion using the GPT model
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    max_tokens=50,
                    n=1,
                    stop=None,
                    temperature=0.7
                )

                # Get the generated category from the GPT response
                category = response.choices[0].text.strip()

                # Add the category field to the data entry
                data[i]['ERN Category'] = category

        return data