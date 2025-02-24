import time
import numpy as np
import torch
from qubots.base_optimizer import BaseOptimizer

def calculate_percentile(input_values, q):
    return np.percentile(input_values, q)

def rescaled_rank_rewards(current_value, previous_values, q=1):
    Cq = calculate_percentile(previous_values, q)
    if current_value < Cq:
        return 1 - q / 100  # Reward for improvement
    elif current_value > Cq:
        return -(q / 100)   # Penalty for worse solutions
    else:
        return 0  # Neutral for exact matches

def rl_local_search(bitstring, QUBO_matrix, const, time_limit, step_size=5, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    QUBO_tensor = torch.tensor(QUBO_matrix, dtype=torch.float32, device=device)
    const_tensor = torch.tensor(const, dtype=torch.float32, device=device)
    state = torch.tensor(bitstring, dtype=torch.float32, device=device)
    best_state = state.clone()
    best_cost = current_cost = torch.matmul(state, torch.matmul(QUBO_tensor, state)) + const_tensor
    progress_costs = [best_cost.item()]
    cut_values = [best_cost.item()]
    P = 100
    num_bits = len(bitstring)
    bit_flip_counts = np.zeros(num_bits)
    bit_flip_total_rewards = np.zeros(num_bits)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        total_actions = np.sum(bit_flip_counts)
        if total_actions > 0:
            ucb_scores = bit_flip_total_rewards / (bit_flip_counts + 1e-5)
            ucb_scores += np.sqrt(2 * np.log(total_actions + 1) / (bit_flip_counts + 1e-5))
            bits_to_flip = np.argpartition(ucb_scores, -step_size)[-step_size:]
        else:
            bits_to_flip = np.random.choice(num_bits, size=step_size, replace=False)
        
        new_state = state.clone()
        for bit in bits_to_flip:
            new_state[bit] = 1 - new_state[bit]
        
        new_cost = torch.matmul(new_state, torch.matmul(QUBO_tensor, new_state)) + const_tensor
        progress_costs.append(new_cost.item())
        cut_values.append(new_cost.item())
        cut_values = cut_values[-P:] if len(cut_values) > P else cut_values
        
        # Greedy acceptance
        if new_cost < current_cost:
            state = new_state.clone()
            current_cost = new_cost.clone()
            if new_cost < best_cost:
                best_state = new_state.clone()
                best_cost = new_cost.clone()
        
        # Update rewards for all flipped bits
        reward = rescaled_rank_rewards(new_cost.item(), cut_values)
        for bit in bits_to_flip:
            bit_flip_total_rewards[bit] += reward
            bit_flip_counts[bit] += 1
        
        if verbose:
            print(f"Current cost: {new_cost.item()}, Best cost: {best_cost.item()}")

    return best_state.cpu().numpy().tolist(), best_cost.cpu().item(), progress_costs

class RLLocalSearchOptimizer(BaseOptimizer):
    def __init__(self, time_limit=5, step_size=5, verbose=False):
        self.time_limit = time_limit
        self.step_size = step_size
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None, **kwargs):
        QUBO_dict = problem.get_qubo()
        max_index = max(max(i, j) for i, j in QUBO_dict.keys())
        n = max_index + 1
        
        if initial_solution is None:
            x = np.random.randint(0, 2, size=n)
        else:
            x = np.array(initial_solution)
        
        QUBO_matrix = np.zeros((n, n))
        for (i, j), value in QUBO_dict.items():
            QUBO_matrix[i, j] = value
            if i != j:
                QUBO_matrix[j, i] = value
                
        best_solution, best_cost, _ = rl_local_search(
            x, QUBO_matrix, 0, self.time_limit, self.step_size, self.verbose
        )
        return best_solution, best_cost