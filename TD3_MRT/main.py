import time
from argparse import ArgumentParser
import numpy as np
import torch
import os
from Trainer import Trainer
import gym
from utils import eval_policy
from replaybuffer import ReplayBuffer


def parse_args():
    parser = ArgumentParser(description='train args')
    #parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-en', '--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('-mt', '--max_timesteps', type=int, default=1e6)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-expn', '--expl_noise', type=bool, default=True)
    # parser.add_argument('-mrt_rm_var', '--mrt_rm_var', type=str)
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-pbts', '--perturbations', nargs='+', help='<Required> perturbation list', required=True)
    parser.add_argument("--mrt", action='store_true')
    parser.add_argument("--mrt_interval", type=int, default=250)
    parser.add_argument('--task', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #torch.cuda.set_device(args.gpu)
    torch.set_num_threads(1)
    ENV_NAME = args.env
    data_dir = os.path.expanduser("~") + "/experiments/RRS_"+"_".join(args.perturbations)
    if args.mrt:
        post_fix = '_mrt_' + str(args.mrt_interval)
    if args.task:
        post_fix = post_fix+'_'+args.task
    data_dir = data_dir+post_fix
    os.makedirs(data_dir, exist_ok=True)
    alias = f'{args.env}_seed_{args.seed}'
    f = open(data_dir+'/{}'.format(alias),'w')
    log = ['eval_reward', 'expl_reward', 'value', 'be_error']
    f.write(",".join(log) + '\n')

    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    eval_env.seed(seed+100)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    args_policy_noise = 0.2
    args_noise_clip = 0.5
    args_policy_freq = 2
    args_max_timesteps = args.max_timesteps
    args_expl_noise = 0.1 if args.expl_noise else 0.0
    args_batch_size = args.batch_size
    args_eval_freq = 1000
    args_start_timesteps = 25000
    perturbations = np.array(args.perturbations).astype(float)
    len_of_ptb = len(perturbations)

    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": 0.99,
              "tau": 0.005, "ptbs": perturbations,
              "policy_noise": args_policy_noise * max_action, "noise_clip": args_noise_clip * max_action,
              "policy_freq": args_policy_freq,'mrt':args.mrt,'mrt_interval':args.mrt_interval}

    # Target policy smoothing is scaled wrt the action scale
    policy = Trainer(**kwargs)
    buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, eval_env)]
    # fill_buffer(buffer, args_start_timesteps, env)

    state = env.reset()
    rewards = []
    expl_rew = 0
    str_time = time.time()
    q_idx = np.random.randint(0, len_of_ptb)
    value_timestep = 0
    update = False
    update_time=0
    value = None
    be_error = None

    for t in range(int(args_max_timesteps)):

        if t < args_start_timesteps:
            action = np.random.uniform(-max_action, max_action, action_dim)
        else:
            action = (
                    policy.select_action(state)
                    + np.random.normal(0, max_action * args_expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)

        rewards.append(reward)

        done_bool = float(done) if len(rewards)<1000 else 0

        buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        if done:
            state = env.reset()
            expl_rew = sum(rewards)
            value_timestep += len(rewards)
            if value_timestep >= 1000:
                update= True
                value_timestep = 0
                q_idx = np.random.randint(0, len_of_ptb)
                rewards = []

        if (t >= args_start_timesteps and update) or t==args_max_timesteps:

            for i in range(t-args_start_timesteps-policy.total_it):
                value, be_error = policy.train(buffer, args_batch_size, q_idx=q_idx)

            update =False

        if t >= args_start_timesteps and (t + 1) % 1000 == 0:
            eval_rew = eval_policy(policy, env=eval_env)
            cur_time = time.time()
            print(f"Total T: {t + 1:.2e}  Expl: {expl_rew:.2f} Eval: {eval_rew:.2f} Time: {cur_time-str_time:.2f}")
            log = [eval_rew, expl_rew, value, be_error]
            str_time = time.time()
            f.write(",".join(map(str, log)) + '\n')
            f.flush()