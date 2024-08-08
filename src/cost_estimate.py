from prompt_generator import PROMPT_TEMPLATE_IMPROVED

def calc_tokens(prompt):
    return len(prompt.split())

def estimate_cost(num_cases, input_tokens_per_case, output_tokens_per_case, cost_per_input_token, cost_per_output_token):
    total_cost = num_cases * (input_tokens_per_case * cost_per_input_token + output_tokens_per_case * cost_per_output_token)
    return total_cost

RAMEDIS_CASES = 624
LIRICAL_CASES = 370
PUMCH_ADM_CASES = 75
MME_CASES = 40
HHS_CASES = 88

example_output = """['Cerebrocostomandibular syndrome'],"+1: Pierre-Robin Sequence: Consistent with cleft palate, respiratory insufficiency, and feeding difficulties. However, it does not explain atrial septal defect, hyperechogenic kidneys, or posterior rib gap.
+2: DiGeorge Syndrome (22q11.2 deletion syndrome): Consistent with atrial septal defect, cleft palate, feeding difficulties, and respiratory insufficiency. Hyperechogenic kidneys and posterior rib gap are not typical.
+3: VACTERL Association: Consistent with atrial septal defect and renal anomalies. However, the patient does not have all the typical features (e.g., vertebral defects, anal atresia).
+4: CHARGE Syndrome: Consistent with heart defects and feeding difficulties. Other typical features (coloboma, atresia choanae, genital abnormalities, ear abnormalities) are not reported.
+5: Branchio-Oto-Renal (BOR) Syndrome: Consistent with renal anomalies. However, other typical features (branchial cleft anomalies, hearing loss) are not reported."""

ROUGH_AVERAGE_OUTPUT_SIZE = calc_tokens(example_output)

OPUS_COST_PER_INPUT_TOKEN = 15 / 1000000
SONNET_COST_PER_INPUT_TOKEN = 3 / 1000000 # Also 15 for output token, but since it's much smaller it's somewhat negligible
LLAMA3_70B_COST_PER_INPUT_TOKEN = 0.9 / 1000000
GPT4TURBO1106_COST_PER_INPUT_TOKEN = 10 / 1000000
GPT4O_COST_PER_INPUT_TOKEN = 5 / 1000000
GPT4OMINI_COST_PER_INPUT_TOKEN = 0.15 / 1000000

OPUS_COST_PER_OUTPUT_TOKEN = 15 / 1000000
SONNET_COST_PER_OUTPUT_TOKEN = 75 / 1000000 # Also 15 for output token, but since it's much smaller it's somewhat negligible
LLAMA3_70B_COST_PER_OUTPUT_TOKEN = 0.9 / 1000000
GPT4TURBO1106_COST_PER_OUTPUT_TOKEN = 30 / 1000000
GPT4O_COST_PER_OUTPUT_TOKEN = 15 / 1000000
GPT4OMINI_COST_PER_OUTPUT_TOKEN = 0.6 / 1000000



input_token_size  = calc_tokens(PROMPT_TEMPLATE_IMPROVED.format(examples="test", description="test")) + 500 # Adding about 500 for the dscription
                          # Roughly should be 16 examples/the description * average size, so I'll say around 20 but could be higher

print("Estimated cost for Opus model:")
print(estimate_cost(8 * (RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES), input_token_size, ROUGH_AVERAGE_OUTPUT_SIZE, OPUS_COST_PER_INPUT_TOKEN, OPUS_COST_PER_OUTPUT_TOKEN))
print("Estimated cost for Sonnet model:")
print(estimate_cost(8 * (RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES), input_token_size, ROUGH_AVERAGE_OUTPUT_SIZE, SONNET_COST_PER_INPUT_TOKEN, SONNET_COST_PER_OUTPUT_TOKEN))
print("Estimated cost for Llama model:")
print(estimate_cost(8 * (RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES), input_token_size, ROUGH_AVERAGE_OUTPUT_SIZE, LLAMA3_70B_COST_PER_INPUT_TOKEN, LLAMA3_70B_COST_PER_OUTPUT_TOKEN))
print("Estimated cost for GPT4 Turbo model:")
print(estimate_cost(8 * (RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES), input_token_size, ROUGH_AVERAGE_OUTPUT_SIZE, GPT4TURBO1106_COST_PER_INPUT_TOKEN, GPT4TURBO1106_COST_PER_OUTPUT_TOKEN))
print("Estimated cost for GPT4o model:")
print(estimate_cost(8 * (RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES), input_token_size, ROUGH_AVERAGE_OUTPUT_SIZE, GPT4O_COST_PER_INPUT_TOKEN, GPT4O_COST_PER_OUTPUT_TOKEN))
print("Estimated cost for GPT4o Mini model:")
print(estimate_cost(8 * (RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES), input_token_size, ROUGH_AVERAGE_OUTPUT_SIZE, GPT4OMINI_COST_PER_INPUT_TOKEN, GPT4OMINI_COST_PER_OUTPUT_TOKEN))
print("Total estimated cost:")
print(estimate_cost(8 * (RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES), input_token_size, ROUGH_AVERAGE_OUTPUT_SIZE, OPUS_COST_PER_INPUT_TOKEN + SONNET_COST_PER_INPUT_TOKEN + LLAMA3_70B_COST_PER_INPUT_TOKEN + GPT4TURBO1106_COST_PER_INPUT_TOKEN + GPT4O_COST_PER_INPUT_TOKEN + GPT4OMINI_COST_PER_INPUT_TOKEN, OPUS_COST_PER_OUTPUT_TOKEN + SONNET_COST_PER_OUTPUT_TOKEN + LLAMA3_70B_COST_PER_OUTPUT_TOKEN + GPT4TURBO1106_COST_PER_OUTPUT_TOKEN + GPT4O_COST_PER_OUTPUT_TOKEN + GPT4OMINI_COST_PER_OUTPUT_TOKEN))
