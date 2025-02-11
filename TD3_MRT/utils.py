import gym
import numpy as np


def eval_policy(policy, env, eval_episodes=3):

    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        while True:
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            if done:
                break

    avg_reward /= eval_episodes
    return avg_reward

def fill_buffer(buffer, fill_num, env):
    # env = gym.make(env_name)
    # state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    state = env.reset()
    episode_timesteps = 0
    for t in range(fill_num):
        episode_timesteps +=1
        action = np.random.uniform(-max_action, max_action, action_dim)
        # action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < 1000 else 0
        buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        if done:
            episode_timesteps = 0
            env.reset()