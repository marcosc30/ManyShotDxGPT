from prompt_generator import PROMPT_TEMPLATE_IMPROVED

def calc_tokens(prompt):
    return len(prompt.split())

def estimate_cost(num_cases, input_tokens_per_case, output_tokens_per_case, cost_per_token):
    total_tokens_per_case = input_tokens_per_case + output_tokens_per_case
    total_tokens = num_cases * total_tokens_per_case
    total_cost = total_tokens * cost_per_token
    return total_cost

RAMEDIS_CASES = 624
LIRICAL_CASES = 370
PUMCH_ADM_CASES = 75
MME_CASES = 40
HHS_CASES = 88

ROUGH_AVERAGE_OUTPUT_SIZE = 3

OPUS_COST_PER_TOKEN = 15 / 1000000
SONNET_COST_PER_TOKEN = 3 / 1000000 # Also 15 for output token, but since it's much smaller it's somewhat negligible
LLAMA3_70B_COST_PER_TOKEN = 0.9 / 1000000
GPT4TURBO1106_COST_PER_TOKEN = 10 / 1000000
GPT4O_COST_PER_TOKEN = 5 / 1000000
GPT4OMINI_COST_PER_TOKEN = 0.15 / 1000000

token_size  = calc_tokens(PROMPT_TEMPLATE_IMPROVED.format(examples="test", description="test")) + 500 # Adding about 500 for the dscription
                          # Roughly should be 16 examples/the description * average size, so I'll say around 20 but could be higher

print("Estimated cost for Opus model:")
print(estimate_cost(RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES, token_size, ROUGH_AVERAGE_OUTPUT_SIZE, OPUS_COST_PER_TOKEN))
print("Estimated cost for Sonnet model:")
print(estimate_cost(RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES, token_size, ROUGH_AVERAGE_OUTPUT_SIZE, SONNET_COST_PER_TOKEN))
print("Estimated cost for Llama model:")
print(estimate_cost(RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES, token_size, ROUGH_AVERAGE_OUTPUT_SIZE, LLAMA3_70B_COST_PER_TOKEN))
print("Estimated cost for GPT4 Turbo model:")
print(estimate_cost(RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES, token_size, ROUGH_AVERAGE_OUTPUT_SIZE, GPT4TURBO1106_COST_PER_TOKEN))
print("Estimated cost for GPT4o model:")
print(estimate_cost(RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES, token_size, ROUGH_AVERAGE_OUTPUT_SIZE, GPT4O_COST_PER_TOKEN))
print("Estimated cost for GPT4o Mini model:")
print(estimate_cost(RAMEDIS_CASES + LIRICAL_CASES + PUMCH_ADM_CASES + MME_CASES + HHS_CASES, token_size, ROUGH_AVERAGE_OUTPUT_SIZE, GPT4OMINI_COST_PER_TOKEN))

