def calc_tokens(prompt):
    return 

def estimate_cost(num_cases, input_tokens_per_case, output_tokens_per_case, cost_per_token):
    total_tokens_per_case = input_tokens_per_case + output_tokens_per_case
    total_tokens = num_cases * total_tokens_per_case
    total_cost = total_tokens * cost_per_token
    return total_cost