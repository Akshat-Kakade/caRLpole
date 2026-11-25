"""A random agent on CartPole-v1.

This "agent" doesn't learn anything. It picks left or right at random.
I just wanna see the RL loop clearly and get a baseline score
thats what my real agents will have to beat.
"""

import gymnasium as gym

NUM_EPISODES = 5

# The environment. render_mode=None to run without a window (much faster).
env = gym.make("CartPole-v1", render_mode="human")

print("Observation space:", env.observation_space)
print("Action space:     ", env.action_space)

for episode in range(NUM_EPISODES):
    # Start a new episode. reset() returns the first state plus an info dict.
    state, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        # AGENT: choose an action
        action = env.action_space.sample()  # random: 0 (push left) or 1 (push right)

        # ENVIRONMENT: apply the action, return what happened
        state, reward, terminated, truncated, info = env.step(action)

        # keep track of this episode's total reward.
        total_reward += reward

        # decide whether the episode is over
        done = terminated or truncated

    print(f"Episode {episode}: total reward = {total_reward}")

env.close()
