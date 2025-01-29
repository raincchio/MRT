import argparse
import os
import time

import gym
import numpy as np
import torch

import TD7_MRT as TD7

# import TR as TD7

def train_online(RL_agent, env, eval_env, args, f):

	start_time = time.time()
	allow_train = False
	value = None
	tdd = None
	last_expl_reward = None
	state, ep_finished = env.reset(), False
	ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

	for t in range(int(args.max_timesteps+1)):
		
		if allow_train:
			action = RL_agent.select_action(np.array(state))
		else:
			action = env.action_space.sample()

		next_state, reward, ep_finished, _ = env.step(action) 
		
		ep_total_reward += reward
		ep_timesteps += 1

		done = float(ep_finished) if ep_timesteps < env._max_episode_steps else 0
		RL_agent.replay_buffer.add(state, action, next_state, reward, done)

		state = next_state

		if allow_train and not args.use_checkpoints:
			value, tdd = RL_agent.train()

		if allow_train and t % args.eval_freq == 0:

			maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, args, last_expl_reward, value, tdd, f)

		if t >= args.timesteps_before_training:
			allow_train = True

		if ep_finished:
			if allow_train and args.use_checkpoints:
				RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

			state, done = env.reset(), False
			last_expl_reward = ep_total_reward
			ep_total_reward, ep_timesteps = 0, 0
			ep_num += 1

def train_offline(RL_agent, env, eval_env, args):
	RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))

	evals = []
	start_time = time.time()

	for t in range(int(args.max_timesteps+1)):
		maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=True)
		RL_agent.train()


def maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, args, expl, value, tdd, f, d4rl=False):
	# if t % args.eval_freq == 0 and allow_train:
	print("-----------------------------------------------------------------")
	print(f"Evaluation at {t} time steps")
	print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

	total_reward = np.zeros(args.eval_eps)
	for ep in range(args.eval_eps):
		state, done = eval_env.reset(), False
		while not done:
			action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
			state, reward, done, _ = eval_env.step(action)
			total_reward[ep] += reward
	eval = total_reward.mean()
	print(f"Average over {args.eval_eps} eval_reward: {eval:.3f} expl_reward: {expl:.3f} value: {value:.3f} be_error: {tdd:.3f} ")
	if d4rl:
		total_reward = eval_env.get_normalized_score(total_reward) * 100
		print(f"D4RL score: {total_reward.mean():.3f}")



	log = [eval, expl, value, tdd]
	f.write(",".join(map(str, log)) + '\n')
	f.flush()
		# dd = np.load(f"{home_directory}/experiments/mrt/{args.file_name}.npy")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	torch.set_num_threads(1)
	# RL
	parser.add_argument("--env", default="Humanoid-v4", type=str)
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--mrt", action='store_true')
	parser.add_argument("--mrt_interval", default=1000, type=int)
	parser.add_argument("--offline", default=False, action=argparse.BooleanOptionalAction)
	parser.add_argument('--use_checkpoints', default=False, action=argparse.BooleanOptionalAction)
	# Evaluation
	parser.add_argument("--timesteps_before_training", default=25e3, type=int)
	parser.add_argument("--eval_freq", default=1e3, type=int)
	parser.add_argument("--eval_eps", default=5, type=int)
	parser.add_argument("--max_timesteps", default=1e6, type=int)
	# File
	parser.add_argument('--file_name', default=None)
	parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
	args = parser.parse_args()
	
	if args.offline:
		import d4rl
		d4rl.set_dataset_path(args.d4rl_path)
		args.use_checkpoints = False

	file_name = f"{args.env}_seed_{args.seed}"

	if args.mrt:
		algo_dir = 'td7_mrt_'+str(args.mrt_interval)
	else:
		algo_dir = 'td7'

	data_dir = os.path.expanduser("~")+"/experiments/" +algo_dir
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	f = open(data_dir + '/{}'.format(file_name), 'w')
	log = ['eval_reward', 'expl_reward', 'value', 'be_error']
	f.write(",".join(log) + '\n')

	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	print("---------------------------------------")
	print(f"Algorithm: TD7, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env.seed(args.seed)
	env.action_space.seed(args.seed)
	eval_env.seed(args.seed+100)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	RL_agent = TD7.Agent(state_dim, action_dim, max_action, args.offline, mrt=args.mrt, mrt_interval=args.mrt_interval)

	if args.offline:
		train_offline(RL_agent, env, eval_env, args)
	else:
		train_online(RL_agent, env, eval_env, args, f)
