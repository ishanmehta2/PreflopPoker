import random
import pickle
from pokerSim import get_global_rewards

q_table = {}

# Initialize belief states for opponent strategies
belief_states = {
    "SB": {"aggressive": 0.3, "passive": 0.5, "balanced": 0.2},
    "BB": {"aggressive": 0.2, "passive": 0.6, "balanced": 0.2},
    "UTG": {"aggressive": 0.1, "passive": 0.7, "balanced": 0.2},
    "MP": {"aggressive": 0.2, "passive": 0.5, "balanced": 0.3},
    "CO": {"aggressive": 0.4, "passive": 0.3, "balanced": 0.3},
}

def get_q_value(state, action):
    """Retrieve the Q-value for a state-action pair."""
    return q_table.get(state, {}).get(action, 0.0)

def update_q_value(state, action, value):
    """Update the Q-value for a state-action pair."""
    if state not in q_table:
        q_table[state] = {}
    q_table[state][action] = value

def save_q_table(filename="q_table.pkl"):
    """Save the Q-table to a file."""
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)

def load_q_table(filename="q_table.pkl"):
    """Load the Q-table from a file."""
    global q_table
    with open(filename, "rb") as f:
        q_table = pickle.load(f)

def update_belief_state(position, observed_action):
    """
    Update the belief state based on observed opponent action.
    :param position: The position of the opponent (e.g., "SB", "BB").
    :param observed_action: The action taken by the opponent ("fold", "call", "raise").
    """
    global belief_states
    if position not in belief_states:
        return  # No belief state for this position

    # Update belief probabilities based on observed action
    for strategy in belief_states[position]:
        if strategy == "aggressive" and observed_action == "raise":
            belief_states[position][strategy] += 0.1  # Increase aggressive probability
        elif strategy == "passive" and observed_action in ["call", "fold"]:
            belief_states[position][strategy] += 0.1  # Increase passive probability
        else:
            belief_states[position][strategy] -= 0.05  # Decrease probabilities slightly

    # Normalize to ensure probabilities sum to 1
    total = sum(belief_states[position].values())
    belief_states[position] = {k: v / total for k, v in belief_states[position].items()}

def sample_opponent_strategy(position):
    """
    Sample an opponent strategy based on the belief state.
    :param position: The position of the opponent (e.g., "SB", "BB").
    :return: A sampled strategy ("aggressive", "passive", "balanced").
    """
    strategies = list(belief_states[position].keys())
    probabilities = list(belief_states[position].values())
    return random.choices(strategies, probabilities)[0]

def calculate_reward(action, global_reward, pot_size, opponent_strategy):
    """
    Reward calculation with belief states and opponent strategy.
    :param action: The action taken ("fold", "check", "call", "raise").
    :param global_reward: The global reward associated with the current hand.
    :param pot_size: The current pot size.
    :param opponent_strategy: The sampled opponent strategy ("aggressive", "passive", "balanced").
    :return: The calculated reward.
    """
    future_potential = 0.1 * pot_size if action.startswith("raise") else 0

    if action == "fold":
        return -0.1 * pot_size
    elif action == "call":
        call_risk = 0.2 * pot_size if opponent_strategy == "aggressive" else 0.1 * pot_size
        return global_reward - call_risk + future_potential
    elif action.startswith("raise"):
        raise_success_chance = 0.3 if opponent_strategy == "passive" else 0.2
        bet_size = int(action.split("_")[1])  # Extract bet size
        return (global_reward + bet_size * raise_success_chance) - 0.3 * pot_size + future_potential
    elif action == "check":
        return global_reward * 0.8

    return 0

def q_learning(global_rewards, num_trials=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Perform Q-learning with belief states and reward shaping.
    """
    for _ in range(num_trials):
        # Initialize a random starting state
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
        pot_size = random.randint(50, 500)
        state = (hand_key, position)

        # Get the global reward for the current hand
        global_reward = global_rewards[hand_key][0]

        # Sample an opponent strategy based on belief state
        opponent_strategy = sample_opponent_strategy(position)

        # Choose an action using epsilon-greedy policy
        if random.random() < epsilon:
            action = random.choice(["check", "call", "fold", "raise_100", "raise_200", "raise_300"])  # Explore
        else:
            q_values = {a: get_q_value(state, a) for a in ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]}
            action = max(q_values, key=q_values.get, default="check")  # Exploit

        # Simulate observed opponent action and update beliefs
        observed_action = random.choice(["fold", "call", "raise"])  # Simulated opponent action
        update_belief_state(position, observed_action)

        # Calculate reward based on action, context, and sampled strategy
        reward = calculate_reward(action, global_reward, pot_size, opponent_strategy)

        # Transition to the next state (remains unchanged for preflop logic)
        next_state = state

        # Update Q-value
        max_next_q = max(get_q_value(next_state, a) for a in ["check", "call", "fold", "raise_100", "raise_200", "raise_300"])
        current_q = get_q_value(state, action)
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        update_q_value(state, action, new_q)

def get_optimal_action(state):
    """Retrieve the optimal action for a given state based on the Q-table."""
    q_values = q_table.get(state, {})
    return max(q_values, key=q_values.get, default="check")

def print_sorted_q_table():
    """Print the Q-table with states sorted by actions alphabetically."""
    print("Q-table:")
    for state, actions in q_table.items():
        sorted_actions = dict(sorted(actions.items()))
        print(f"{state}: {sorted_actions}")

def evaluate_policy(global_rewards, num_trials=100):
    """
    Evaluate the policy by simulating episodes with the learned Q-table.
    """
    total_reward = 0
    for _ in range(num_trials):
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
        pot_size = random.randint(50, 500)
        state = (hand_key, position)
        
        opponent_strategy = sample_opponent_strategy(position)
        action = get_optimal_action(state)
        global_reward = global_rewards[hand_key][0]
        reward = calculate_reward(action, global_reward, pot_size, opponent_strategy)
        total_reward += reward
    return total_reward / num_trials

# Simulate Q-learning
global_rewards = get_global_rewards()

q_learning(global_rewards, num_trials=5000)
print_sorted_q_table()

# Test the optimal policy for a specific state
test_state = ("AA", "CO")
optimal_action = get_optimal_action(test_state)
print(f"Optimal action for state {test_state}: {optimal_action}")

# Evaluate the policy
average_reward = evaluate_policy(global_rewards, num_trials=100)
print(f"Average reward during evaluation: {average_reward}")
