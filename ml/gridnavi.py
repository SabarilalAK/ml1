import numpy as np
import matplotlib.pyplot as plt

# Grid and Q-learning parameters
grid_size = 5
num_episodes = 500
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
actions = [0, 1, 2, 3]  # Actions: 0=up, 1=down, 2=left, 3=right
q_table = np.zeros((grid_size, grid_size, len(actions)))

# Goal and obstacles in the environment
goal = (4, 4)
obstacles = [(1, 1), (2, 1), (3, 3)]
rewards = np.zeros((grid_size, grid_size))
rewards[goal] = 1
for obs in obstacles:
    rewards[obs] = -1

# Training with Q-learning
for episode in range(num_episodes):
    position = (0, 0)  # Start position
    while position != goal:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)  # Explore
        else:
            action = np.argmax(q_table[position[0], position[1]])  # Exploit

        # Determine new position based on action
        new_position = position
        if action == 0 and position[0] > 0:  # Move up
            new_position = (position[0] - 1, position[1])
        elif action == 1 and position[0] < grid_size - 1:  # Move down
            new_position = (position[0] + 1, position[1])
        elif action == 2 and position[1] > 0:  # Move left
            new_position = (position[0], position[1] - 1)
        elif action == 3 and position[1] < grid_size - 1:  # Move right
            new_position = (position[0], position[1] + 1)

        # Receive reward and update Q-value
        reward = rewards[new_position]
        q_table[position[0], position[1], action] += learning_rate * (
            reward + discount_factor * np.max(q_table[new_position[0], new_position[1]]) - q_table[position[0], position[1], action]
        )
        
        # Move to the new position
        position = new_position

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training completed.")

# Visualization of Q-values for each action
fig, axs = plt.subplots(1, len(actions), figsize=(15, 5))
for i in range(len(actions)):
    im = axs[i].imshow(q_table[:, :, i], cmap='hot', interpolation='nearest')
    axs[i].set_title(f'Q-values for Action {i}')
    axs[i].set_xticks(np.arange(grid_size))
    axs[i].set_yticks(np.arange(grid_size))
    axs[i].set_xticklabels(np.arange(grid_size))
    axs[i].set_yticklabels(np.arange(grid_size))
    axs[i].grid(False)

plt.suptitle('Q-values for Each Action in the Grid Environment')
plt.tight_layout(rect=[0, 0, 1, 0.96])
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical', label='Q-value')
plt.show()
