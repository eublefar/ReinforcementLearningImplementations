import argparse

import gym
import pybullet_envs
import torch
from gym import wrappers, logger
from sys import argv
import logging

import importlib

from tensorboardX import SummaryWriter


def main(args):
    kwargs = {}
    if args.render_env:
        kwargs['render'] = args.render_env
        print('rendering...')

    if args.log_dir == 'polyaxon':
        from polyaxon_helper import get_outputs_path
        args.log_dir = get_outputs_path()
        print(f'Writing to logdir: {args.log_dir}')
    writer = SummaryWriter(log_dir=args.log_dir)

    env = gym.make(args.env, **kwargs)
    env.seed(args.random_seed)

    agent_class = getattr(importlib.import_module(args.agent_module), args.agent)
    agent = agent_class(env.observation_space, env.action_space, argv, writer)

    try:
        train(args, agent, writer, env)
    except KeyboardInterrupt:
        env.close()
        agent.save('Interrupt')
        raise KeyboardInterrupt
    agent.save('Final')
    env.close()


def train(args, agent, writer, env):
    # random loop
    for i in range(args.random_episodes):
        ob = env.reset()
        for _ in range(args.max_episode_len):
            ob, reward, done = step_random(env, agent, ob)
            if done:
                break

    if args.use_monitor:
        if args.monitor_dir == 'polyaxon':
            from polyaxon_helper import get_outputs_path
            args.monitor_dir = get_outputs_path()
            print(f'Using monitor_dir: {args.monitor_dir}')
        print(f"Using Gym monitor to save videos : {args.use_gym_monitor} 123 {args.render_env}")
        env = wrappers.Monitor(env, directory=args.monitor_dir, force=True)

    # policy loop
    global_step = 0
    for i in range(args.random_episodes, args.max_episodes):
        ob = env.reset()
        reward_per_ep = 0
        for ep_step in range(args.max_episode_len):
            global_step += 1
            ob, reward, done = step_policy(env, agent, ob, i)
            reward_per_ep += reward
            if done:
                break
        writer.add_scalar("reward", reward_per_ep, global_step=i)
        writer.add_scalar("avg_legth", ep_step, global_step=i)

        if i % args.checkpoint_episodes:
            agent.save('checkpoint_{}'.format(i))

    writer.close()
    env.close()


def step_random(env, agent, last_ob, episode_num):
    action = env.action_space.sample()
    ob, reward, done, _ = env.step(action)
    agent.memorize(last_ob, action, reward, done, ob)
    return ob, reward, done


def step_policy(env, agent, last_ob, episode_num):
    state = torch.from_numpy(last_ob).view(1, -1).float()
    action = agent.act(state, episode_num)
    ob, reward, done, _ = env.step(action)
    agent.memorize(last_ob, action, reward, done, ob)
    agent.update(episode_num)
    return ob, reward, done

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    logger = logging.getLogger('global')
    logger.setLevel(logging.INFO)
    # run parameters

    parser.add_argument('--use-monitor', help='record gym results', action='store_true')
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--agent-module', help='Agent module name to use.', default='agents.DDPGAgent')
    parser.add_argument('--agent', help='Agent class name to use.', default='DDPGAgent')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234, type=int)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000, type=int)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000, type=int)
    parser.add_argument('--random-episodes', help='number of episodes to sample randomly', default=20, type=int)
    parser.add_argument('--checkpoint-episodes', help='number of episodes to save model', default=1000, type=int)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--log-dir', help='directory for storing tensorboard summary', default='./results/tensorboard_data')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args, _ = parser.parse_known_args()

    print(f'Parsed global parameters {args}')

    main(args)
