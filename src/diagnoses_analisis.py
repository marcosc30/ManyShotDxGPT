import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from numpy import NaN

# Load the environment variables from the .env file
load_dotenv()

# Delete entries with no GT, I should've done this earlier honestly

# LLM Version from original paper
# # Initialize the AzureChatOpenAI model
# model = AzureChatOpenAI(
#     openai_api_version = str(os.getenv("OPENAI_API_VERSION")),
#     deployment_name="nav29",
#     temperature=0,
#     # request_timeout=128,
#     max_tokens=800
# )

# gpt-4o
model_name = "gpt-4o"
openai_api_key=os.getenv("OPENAI_API_KEY")
gpt4o = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model_name,
        temperature = 0,
        max_tokens = 800,
    )

def get_scores(model, dataset, model_tested, many_shot, cat, i=False):
    if many_shot:
        many_shot = 'manyshot'
    else:
        many_shot = 'noshot'
    if cat:
        cat = 'cat'
    else:
        cat = 'nocat'
    if i:
        i = 'i'
    else:
        i = 'ni'
    # Load the diagnoses data
    input_path = f'data/Diagnoses/{model_tested}/diagnoses_{dataset}_{model_tested}_{many_shot}_{cat}_{i}.csv'
    df = pd.read_csv(input_path)

    # Summarize the data
    # print(df.describe())
    # print(df.head())
    # print(df.shape)

    # Now we will analyze the data to see if the GT is in the column 1 of diagnoses (each column has between 3 to 6 diagnoses), between P1 and P5 predictions.
    # Then we will create a new DataFrame to store the scores of the predictions.
    # We will iterate over the rows in the diagnoses data and we will compare the GT with the predictions column 1 first.
    # If the GT is in the predictions column 1, we will store the score as P1 if the first prediction is the GT, and P5 otherwise.
    # Create a new DataFrame to store the scores of the predictions
    scores_df = pd.DataFrame(columns=['GT', 'Score'])

    PROMPT_TEMPLATE = """Behave like a medical doctor reviewing patient diagnoses. You will be given a Ground Truth diagnosis (GT) and 5 Predicted diagnoses (P1-P5). Compare the GT to the predictions and return a classification: 

    If GT exactly matches P1, return "P1".  
    If GT is contained within or is a broader term for P1-P5, return "P5".
    If GT does not match any of P1-P5, return "P0".

    The GT may be a more general diagnosis, while predictions may include specific conditions. Broadly match GT to any prediction it reasonably encompasses.
    ----------------------------------------
    The text is:

    GT: {gt}

    Predictions:

    {predictions} 
    ----------------------------------------
    Return either "P1", "P5", or "P0". Do not return any other text.
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    # Iterate over the rows in the diagnoses data
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Get the ground truth (GT) and the first prediction
        if row['GT'] == "[]":
            continue

        gt = row['GT']
        predictions = row['Diagnosis 1']
                          
        # Generate a score for the prediction
        formatted_prompt = chat_prompt.format_messages(gt=gt, predictions=predictions)
        #print(formatted_prompt)
        attempts = 0
        while attempts < 2:
            try:
                score = model(formatted_prompt).content
                break
            except Exception as e:
                attempts += 1
                print(e)
                if attempts == 2:
                    score = "P0"

        print(f"GT: {gt}, Score: {score}")

        # Add the score to the new DataFrame
        scores_df.loc[index] = [gt, score]

    # Save the scores to a new CSV file
    output_file = f'scores_{dataset}_{model_tested}_{many_shot}_{cat}_{i}.csv'
    output_path = f'data/Scores/{model_tested}/{output_file}'
    scores_df.to_csv(output_path, index=False)


def get_all_scores (model, model_tested, datasets=['PUMCH_ADM','RAMEDIS', 'LIRICAL', 'MME', 'HMS']):
    for dataset in datasets:
        get_scores(model, dataset, model_tested, many_shot=True, cat=True, i=False)
        get_scores(model, dataset, model_tested, many_shot=False, cat=True, i=False)
        if dataset == 'RAMEDIS' or dataset == 'PUMCH_ADM':
            get_scores(model, dataset, model_tested, many_shot=True, cat=False, i=False)
            get_scores(model, dataset, model_tested, many_shot=False, cat=False, i=False)


#get_all_scores(gpt4o, "gpt4omini")
get_all_scores(gpt4o, "gpt4o")
get_all_scores(gpt4o, "c3opus")
get_all_scores(gpt4o, "c3sonnet")

    






