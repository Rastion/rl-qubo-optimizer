import time
import numpy as np
import torch
import random
from qubots.base_optimizer import BaseOptimizer

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

def calculate_percentile(input_values, q):
    return np.percentile(input_values, q)

def rescaled_rank_rewards(current_value, previous_values, q=1):
    Cq = calculate_percentile(previous_values, q)
    print(f"Percentile cost: {Cq}")
    if current_value < Cq:
        return -(q / 100)
    elif current_value > Cq:
        return 1 - q / 100
    else:
        return 1 if random.random() > 0.5 else -1

def simulated_annealing(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return True
    else:
        prob_accept = np.exp(-(new_cost - current_cost) / temperature)
        return np.random.rand() < prob_accept

def rl_local_search(bitstring, QUBO_matrix, const, time_limit, temperature=10, verbose=False):
    """
    RL local search algorithm for QUBO problems.
    It takes an initial candidate (a binary vector), the QUBO matrix and constant,
    and then iteratively flips bits based on a simple UCB-inspired rule and simulated annealing.
    Returns:
      - best_state: the best candidate found (as a list of 0s and 1s)
      - best_cost: the corresponding cost
      - progress_costs: a list of cost values at each iteration
    """
    # Set up torch tensors on the available device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    QUBO_tensor = torch.tensor(QUBO_matrix, dtype=torch.float32, device=device)
    const_tensor = torch.tensor(const, dtype=torch.float32, device=device)
    state = torch.tensor(bitstring, dtype=torch.float32, device=device)
    best_state = state.clone()
    best_cost = torch.matmul(state, torch.matmul(QUBO_tensor, state)) + const_tensor
    progress_costs = [best_cost.item()]
    print("Initial cost :", best_cost.item())
    cut_values = [best_cost.item()]
    P = 100
    num_bits = len(bitstring)
    bit_flip_counts = np.zeros(num_bits)
    bit_flip_total_rewards = np.zeros(num_bits)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        iteration_start = time.time()
        total_actions = np.sum(bit_flip_counts)
        if total_actions > 0:
            ucb_scores = bit_flip_total_rewards / (bit_flip_counts + 1e-5)
            ucb_scores += np.sqrt(2 * np.log(total_actions) / (bit_flip_counts + 1e-5))
            bit_to_flip = np.argmax(ucb_scores)
        else:
            bit_to_flip = np.random.choice(num_bits)
        new_state = state.clone()
        new_state[bit_to_flip] = 1 - new_state[bit_to_flip]
        new_cost = torch.matmul(new_state, torch.matmul(QUBO_tensor, new_state)) + const_tensor
        progress_costs.append(new_cost.item())
        if simulated_annealing(best_cost.item(), new_cost.item(), temperature):
            if new_cost.item() < best_cost.item():
                best_state = new_state.clone()
                best_cost = new_cost.clone()
            state = new_state.clone()
            cut_values.append(new_cost.item())
            if len(cut_values) > P:
                cut_values = cut_values[-P:]
        bit_flip_total_rewards[bit_to_flip] += rescaled_rank_rewards(new_cost.item(), cut_values)
        bit_flip_counts[bit_to_flip] += 1
        if verbose:
            iter_time = time.time() - iteration_start
            print(f"Iteration time: {iter_time:.4f} sec, Current cost: {new_cost.item()}, Best cost: {best_cost.item()}")
    return best_state.cpu().numpy().tolist(), best_cost.cpu().item(), progress_costs

class RLLocalSearchOptimizer(BaseOptimizer):
    """
    An RL-based local search optimizer for QUBO problems.
    
    This optimizer assumes that the problem instance implements:
      - random_solution(): returns an initial binary vector.
      - get_qubo(): returns (QUBO_matrix, qubo_constant).
    """
    def __init__(self, time_limit=5, temperature=10, verbose=False):
        self.time_limit = time_limit      # Time (in seconds) to run the RL search.
        self.temperature = temperature    # Temperature parameter for simulated annealing.
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None, **kwargs):

        

        # Extract QUBO dictionary from problem
        QUBO_dict = problem.get_qubo()

        # Find the matrix size
        max_index = max(max(i, j) for i, j in QUBO_dict.keys())  # Determine size from dictionary keys
        n = max_index + 1  # Since indices are 0-based

        if initial_solution is None:
            x = np.random.randint(0, 2, size=n)

        # Initialize the QUBO matrix as an n x n NumPy array
        QUBO_matrix = np.zeros((n, n))

        # Populate the matrix from the dictionary
        for (i, j), value in QUBO_dict.items():
            QUBO_matrix[i, j] = value
            if i != j:
                QUBO_matrix[j, i] = value  # Ensure symmetry if needed
                
        # Run the RL local search.
        best_solution, best_cost, progress = rl_local_search(
            x,
            QUBO_matrix,
            0,
            time_limit=self.time_limit,
            temperature=self.temperature,
            verbose=self.verbose
        )
        return best_solution, best_cost