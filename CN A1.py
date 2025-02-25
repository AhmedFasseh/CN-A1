import math

def generate_random(low, high, seed):
    seed = (seed * 9301 + 49297) % 233280
    return low + (seed / 233280.0) * (high - low)

def calculate_exponential(x, terms=10):
    result, factorial, power = 1, 1, 1
    for i in range(1, terms):
        factorial *= i
        power *= x
        result += power / factorial
    return result

def tanh_activation_taylor(x):
    return (calculate_exponential(x) - calculate_exponential(-x)) / (calculate_exponential(x) + calculate_exponential(-x))

def tanh_activation_math(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def compute_outputs(input1, input2, params, bias_values, activation_fn):
    hidden_neuron1 = activation_fn(input1 * params['w_a'] + input2 * params['w_b'] + bias_values['bias_h'])
    hidden_neuron2 = activation_fn(input1 * params['w_c'] + input2 * params['w_d'] + bias_values['bias_h'])
    output1 = activation_fn(hidden_neuron1 * params['w_e'] + hidden_neuron2 * params['w_f'] + bias_values['bias_o'])
    output2 = activation_fn(hidden_neuron1 * params['w_g'] + hidden_neuron2 * params['w_h'] + bias_values['bias_o'])

    return output1, output2, hidden_neuron1, hidden_neuron2

seed = 42
weights_random = [generate_random(-0.5, 0.5, seed + i) for i in range(8)]

params_random = {
    'w_a': weights_random[0], 'w_b': weights_random[1],
    'w_c': weights_random[2], 'w_d': weights_random[3],
    'w_e': weights_random[4], 'w_f': weights_random[5],
    'w_g': weights_random[6], 'w_h': weights_random[7]
}

params_fixed = {
    'w_a': 0.15, 'w_b': 0.20,
    'w_c': 0.25, 'w_d': 0.30,
    'w_e': 0.40, 'w_f': 0.45,
    'w_g': 0.50, 'w_h': 0.55
}

bias_values_random = {'bias_h': 0.5, 'bias_o': 0.7}
bias_values_fixed = {'bias_h': 0.35, 'bias_o': 0.60}

input1, input2 = 0.05, 0.10
target_o1, target_o2 = 0.1, 0.99

output_rand1, output_rand2, h_rand1, h_rand2 = compute_outputs(input1, input2, params_random, bias_values_random, tanh_activation_taylor)
output_fixed1, output_fixed2, h_fixed1, h_fixed2 = compute_outputs(input1, input2, params_fixed, bias_values_fixed, tanh_activation_math)

E_o1_rand = 0.5 * (target_o1 - output_rand1) ** 2
E_o2_rand = 0.5 * (target_o2 - output_rand2) ** 2
total_loss_rand = E_o1_rand + E_o2_rand
E_o1_fixed = 0.5 * (target_o1 - output_fixed1) ** 2
E_o2_fixed = 0.5 * (target_o2 - output_fixed2) ** 2
total_loss_fixed = E_o1_fixed + E_o2_fixed

print("=== Random Weights with Taylor Approximation ===")
print(f"Hidden Outputs: h1 = {round(h_rand1, 6)}, h2 = {round(h_rand2, 6)}")
print(f"Output Values: o1 = {round(output_rand1, 6)}, o2 = {round(output_rand2, 6)}")
print(f"Losses: E_o1 = {round(E_o1_rand, 6)}, E_o2 = {round(E_o2_rand, 6)}")
print(f"Total Loss: {round(total_loss_rand, 6)}")
print("\n=== Fixed Weights with math.exp ===")
print(f"Hidden Outputs: h1 = {round(h_fixed1, 6)}, h2 = {round(h_fixed2, 6)}")
print(f"Output Values: o1 = {round(output_fixed1, 6)}, o2 = {round(output_fixed2, 6)}")
print(f"Losses: E_o1 = {round(E_o1_fixed, 6)}, E_o2 = {round(E_o2_fixed, 6)}")
print(f"Total Loss: {round(total_loss_fixed, 6)}")
