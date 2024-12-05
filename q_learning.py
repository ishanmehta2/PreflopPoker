import random
import pickle
from pokerSim import get_global_rewards

# Initialize the Q-table
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
    belief_states[position] = {k: max(v / total, 0.01) for k, v in belief_states[position].items()}  # Avoid zero probabilities

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
        try:
            bet_size = int(action.split("_")[1])  # Extract bet size
        except (IndexError, ValueError):
            bet_size = 100  # Default bet size if parsing fails
        return (global_reward + bet_size * raise_success_chance) - 0.3 * pot_size + future_potential
    elif action == "check":
        return global_reward * 0.8

    return 0

def get_optimal_action(state):
    """
    Retrieve the optimal action based on the current Q-values for a given state.
    :param state: The current state (e.g., ("AA", "CO")).
    :return: The optimal action as a string.
    """
    actions = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]
    q_values = {action: get_q_value(state, action) for action in actions}
    return max(q_values, key=q_values.get, default="check")

def evaluate_policy(global_rewards, num_trials=100):
    """
    Evaluate the current policy by running multiple trials and calculating the average reward.
    :param global_rewards: The global rewards dictionary.
    :param num_trials: The number of trials to run for evaluation.
    :return: The average reward.
    """
    total_reward = 0
    for _ in range(num_trials):
        # Initialize a random starting state
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
        state = (hand_key, position)

        # Sample an opponent strategy based on belief state
        opponent_strategy = sample_opponent_strategy(position)

        # Get the optimal action based on current Q-values
        action = get_optimal_action(state)

        # Get the global reward for the current hand
        global_reward = global_rewards[hand_key][0]

        # Calculate a random pot size for this trial
        pot_size = random.randint(50, 500)

        # Calculate the reward based on action, context, and sampled opponent strategy
        reward = calculate_reward(action, global_reward, pot_size, opponent_strategy)

        # Update the total reward with the reward from this trial
        total_reward += reward

    # Calculate and return the average reward
    average_reward = total_reward / num_trials
    return average_reward

def choose_action(state, epsilon):
    """
    Choose an action based on epsilon-greedy policy.
    :param state: The current state.
    :param epsilon: The probability of choosing a random action.
    :return: The chosen action.
    """
    actions = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]
    if random.random() < epsilon:
        return random.choice(actions)  # Explore
    else:
        return get_optimal_action(state)  # Exploit

def q_learning_training(episodes=10000, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995):
    """
    Train the Q-table using Q-learning.
    :param episodes: Number of training episodes.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    :param epsilon_start: Initial exploration rate.
    :param epsilon_end: Minimum exploration rate.
    :param epsilon_decay: Decay rate for exploration.
    """
    global q_table
    epsilon = epsilon_start
    global_rewards = get_global_rewards()

    for episode in range(1, episodes + 1):
        # Initialize a random starting state
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
        state = (hand_key, position)

        # Choose an action using epsilon-greedy policy
        action = choose_action(state, epsilon)

        # Sample an opponent strategy based on belief state
        opponent_strategy = sample_opponent_strategy(position)

        # Get the global reward for the current hand
        global_reward = global_rewards[hand_key][0]

        # Calculate a random pot size for this trial
        pot_size = random.randint(50, 500)

        # Calculate the reward based on action, context, and sampled opponent strategy
        reward = calculate_reward(action, global_reward, pot_size, opponent_strategy)

        # For simplicity, assume the next state is terminal (single-step)
        next_state = None

        # Get the maximum Q-value for the next state (0 if terminal)
        if next_state is None:
            max_future_q = 0
        else:
            actions_next = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]
            max_future_q = max([get_q_value(next_state, a) for a in actions_next], default=0)

        # Current Q-value
        current_q = get_q_value(state, action)

        # Q-learning update
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        update_q_value(state, action, new_q)

        # Optionally update belief states based on opponent actions (if applicable)
        # For this example, we assume the observed action is the opponent's response
        # Here, we randomly simulate an opponent action based on their strategy
        # You might have a more sophisticated way to observe opponent actions
        opponent_action = simulate_opponent_action(opponent_strategy)
        update_belief_state(position, opponent_action)

        # Decay epsilon
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_end)

        # Optionally log progress
        if episode % 1000 == 0:
            avg_reward = evaluate_policy(global_rewards, num_trials=100)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    # Save the trained Q-table
    save_q_table()

def simulate_opponent_action(strategy):
    """
    Simulate an opponent action based on their strategy.
    :param strategy: Opponent's strategy ("aggressive", "passive", "balanced").
    :return: Opponent's action ("fold", "call", "raise").
    """
    if strategy == "aggressive":
        return random.choices(["raise", "call", "fold"], weights=[0.6, 0.3, 0.1])[0]
    elif strategy == "passive":
        return random.choices(["fold", "call", "raise"], weights=[0.5, 0.4, 0.1])[0]
    else:  # balanced
        return random.choices(["raise", "call", "fold"], weights=[0.3, 0.4, 0.3])[0]

# Load global rewards
global_rewards = get_global_rewards()

# Optional: Load existing Q-table if available
try:
    load_q_table()
    print("Loaded existing Q-table.")
except FileNotFoundError:
    print("No existing Q-table found. Starting fresh.")

# Train the Q-learning agent
print("Starting Q-learning training...")
q_learning_training(episodes=100000)  # Adjust the number of episodes as needed
print("Training completed.")

# Test the optimal policy for a specific state after training
test_state = ("KJo", "MP")
optimal_action = get_optimal_action(test_state)
print(f"Optimal action for state {test_state}: {optimal_action}")

# Evaluate the policy
average_reward = evaluate_policy(global_rewards, num_trials=100)
print(f"Average reward during evaluation: {average_reward:.2f}")

def inspect_co_q_values_with_optimal_actions():
    """
    Inspect and display Q-values for all hand pairings in the Cutoff (CO) position,
    along with the optimal action.
    """
    actions = ["fold", "call", "check", "raise_100", "raise_200", "raise_300"]
    co_states = [state for state in q_table.keys() if state[1] == "SB"]

    # Sort hands for better readability (optional)
    sorted_co_states = sorted(co_states, key=lambda x: x[0])

    print(f"Total CO States in Q-Table: {len(sorted_co_states)}\n")
    header = f"{'Hand':<5} | " + " | ".join([f"{action:<10}" for action in actions]) + " | Optimal Action"
    print(header)
    print("-" * len(header))

    for state in sorted_co_states:
        hand = state[0]
        q_values = {action: get_q_value(state, action) for action in actions}
        q_values_str = [f"{q_values[action]:<10.2f}" for action in actions]
        optimal_action = max(q_values, key=q_values.get, default="check")
        print(f"{hand:<5} | " + " | ".join(q_values_str) + f" | {optimal_action}")

#inspect_co_q_values_with_optimal_actions()
