# This file just contains some different prompts that were tested
# The final function is redundant now as the approach was changed to format it at the moment of formatting the final prompt before diagnosis

# These are the examples
PROMPT_TEMPLATE_RARE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Give me a list of potential rare diseases with a short description. Shows for each potential rare diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms:{description}"

PROMPT_TEMPLATE_ORIG = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Give me a list of potential diseases with a short description. Shows for each potential diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms:{description}"

PROMPT_TEMPLATE_MORE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Continue the list of potential rare diseases without repeating any disease from the list I give you. If you repeat any, it is better not to return it. Shows for each potential rare diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate a short description and which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms: {description}. Each must have this format '\n\n+7.' for each potencial rare diseases. The list is: {initial_list} "

PROMPT_TEMPLATE_IMPROVED = """
<prompt> As an AI-assisted diagnostic tool, your task is to analyze the given patient symptoms and generate a list of the top 5 potential diagnoses. Follow these steps:
Carefully review the patient's reported symptoms.
In the <thinking></thinking> tags, provide a detailed analysis of the patient's condition: a. Highlight any key symptoms or combinations of symptoms that stand out. b. Discuss possible diagnoses and why they might or might not fit the patient's presentation. c. Suggest any additional tests or information that could help narrow down the diagnosis.
In the <top5></top5> tags, generate a list of the 5 most likely diagnoses that match the given symptoms: a. Assign each diagnosis a number, starting from 1 (e.g., "\n\n+1", "\n\n+2", etc.). b. Provide the name of the condition, followed by a colon (":"). c. Indicate which of the patient's symptoms are consistent with this diagnosis. d. Mention any key symptoms of the condition that the patient did not report, if applicable.
Remember:

Do not use "\n\n-" in your response. Only use "\n\n+" when listing the diagnoses.
The <thinking> section should come before the <top5> section.
Use the provided examples to guide your response.
<diagnosis_examples>
{examples}
<\diagnosis_examples>
Patient Symptoms:
<patient_description>
{description} 
</patient_description>
</prompt>
"""

PROMPT_TEMPLATE_IMPROVED_SPLIT1 = """
<prompt> As an AI-assisted diagnostic tool, your task is to analyze the given patient symptoms and generate a list of the top 5 potential diagnoses. Follow these steps:
Carefully review the patient's reported symptoms.
In the <thinking></thinking> tags, provide a detailed analysis of the patient's condition: a. Highlight any key symptoms or combinations of symptoms that stand out. b. Discuss possible diagnoses and why they might or might not fit the patient's presentation. c. Suggest any additional tests or information that could help narrow down the diagnosis.
In the <top5></top5> tags, generate a list of the 5 most likely diagnoses that match the given symptoms: a. Assign each diagnosis a number, starting from 1 (e.g., "\n\n+1", "\n\n+2", etc.). b. Provide the name of the condition, followed by a colon (":"). c. Indicate which of the patient's symptoms are consistent with this diagnosis. d. Mention any key symptoms of the condition that the patient did not report, if applicable.
Remember:

Do not use "\n\n-" in your response. Only use "\n\n+" when listing the diagnoses.
The <thinking> section should come before the <top5> section.
###
Use the provided examples to guide your response.
<diagnosis_examples>
{examples}
<\diagnosis_examples>
###"""


PROMPT_TEMPLATE_IMPROVED_SPLIT2 = """
Patient Symptoms:
<patient_description>
{description} 
</patient_description>
</prompt>
"""

datasets = ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"]
examples = {}
indices = {}


# def setup_prompts(dataset):
#     for ern_category in ern_categories:
#         data = load_dataset('chenxz/RareBench', dataset, split='test', trust_remote_code=True)
#         examples[ern_category], indices[ern_category] = setup_manyshot_ex(data, ern_category)
#     ern_prompts = {}
#     prompt_template = PROMPT_TEMPLATE_IMPROVED_SPLIT1
#     for ern_category in ern_categories:
#         ern_prompts[ern_category] = prompt_template.format(examples=examples[ern_category]) + PROMPT_TEMPLATE_IMPROVED_SPLIT2
#     return ern_prompts

