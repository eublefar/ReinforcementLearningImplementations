import torch
import gym
import pybullet_envs
from DDPGAgent import DDPGAgent
import sys

if __name__=='__main__':
    env = gym.make("MinitaurBulletEnv-v0", render=True)
    agent = DDPGAgent(env.observation_space, env.action_space, sys.argv[1:])

    agent.critic.load_state_dict(torch.load("outputs/root/ReinforcementLearningImplementations/experiments/29/checkpoint/critic"))
    agent.actor.load_state_dict(torch.load("outputs/root/ReinforcementLearningImplementations/experiments/29/checkpoint/actor"))
    def no_noise():
        return 0
    agent.noise = no_noise
    while True:
        state = env.reset()
        done = False
        while not done:
            state, reward, done, _ = env.step(agent.act(torch.from_numpy(state).view(1, -1).float()))
