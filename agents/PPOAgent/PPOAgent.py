from torch import nn
import torch
from argparse import ArgumentParser
from policies.ActorCriticPolicy import ActorCritic
from pathlib import Path
import os
import importlib
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO:
#  - Add logging of everything + animations for functions over time
#  - Add common buffer class with samplers
#  - Add module superclass that will parse parameters it needs


class Episode:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []


class PPOAgent:
    def __init__(self, observation_space, action_space, args_for_parse, summary_writer=None):

        self.summary_writer = summary_writer

        parser = ArgumentParser(description='Deep Deterministic Policy Gradient')
        parser.add_argument('--dkl-penalty', help='Use penalty instead of surrogate objective clipping', action='store_true')
        parser.add_argument('--sampler', help='Sampler class name to use for exploration and learning', type=str)
        parser.add_argument('--lr', help='actor network learning rate', default=0.0001, type=float)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99, type=float)
        parser.add_argument('--batch-size', help='size of update memory', default=100, type=int)
        parser.add_argument('--action-std', help='standard deviation of action output signal', default=0.6, type=float)
        parser.add_argument('--epochs', help='epochs to update per step', default=5, type=int)
        parser.add_argument('--latent', help='Size of a hidden layers', default=64, type=int)
        parser.add_argument('--epsilon', help='clip objective threshold', default=0.2, type=float)
        parser.add_argument('--lam', help='Gae lambda parameter', default=0.5, type=float)
        parser.add_argument('--model-dir', help='directory for storing gym results', default='./saved_models')
        parser.add_argument('--cuda', help='Enable gpu optimization', action='store_true')

        self.args, _ = parser.parse_known_args(args_for_parse)
        print(f'Parsed model parameters {self.args}')

        if self.args.model_dir == 'polyaxon' :
            from polyaxon_helper import get_outputs_path
            self.args.model_dir = get_outputs_path()

        self.gamma = self.args.gamma
        self.action_dim = action_space.shape[0]
        self.state_dim = observation_space.shape[0]
        self.critic_update_weight = 0.7
        self.entropy_update_weight = 0.01

        if any(action_space.high != -action_space.low):
            raise ValueError(f"Env action space is not symmetric. high :{action_space.high} low: {action_space.low}")

        self.action_scale = action_space.high[0]

        self.policy_sampler = agent_class = getattr(
            importlib.import_module('policy_samplers.{}'.format(self.args.sampler)),
            self.args.sampler) (args_for_parse)

        self.policy = ActorCritic(self.state_dim, self.action_dim, self.args.latent, self.args.action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.args.lr,)
        self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.args.latent, self.args.action_std).to(device)
        self.loss = nn.MSELoss(reduce=False)

        self.step = 0
        self.memory = [Episode()]
        self.new_episode = False

    def act(self, state, episode):
        self.policy_old.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.policy_old.act(state)
        action_sampled, logprob = self.policy_sampler.sample_action(action)
        self.memory[-1].states.append(state)
        self.memory[-1].logprobs.append(logprob)
        self.memory[-1].actions.append(action_sampled)
        self.policy_old.train()
        return action_sampled.cpu().data.numpy().flatten() * self.action_scale

    def memorize(self, s, a, r, terminal, s_prim):
        self.memory[-1].rewards.append(r)
        if terminal:
            # self.memory[-1].states.append(torch.FloatTensor(s_prim.reshape(1, -1)).to(device))
            self.new_episode = True

    def build_triangle_matrix(self, v):
        tri = torch.zeros((len(v), len(v)))
        for i in range(len(v)):
            tri[i, i:] = v
            v = v[:-1]
        return tri

    def update(self, episode=0):
        if self.new_episode:
            if episode % self.args.batch_size == 0 and episode:
                self.fit_batch(episode)
                self.memory = []
            self.memory.append(Episode())
            self.new_episode = False

    def fit_batch(self, ep):
        m = []
        for e in self.memory:
            e = self.process_episode(e)
            m.append(e)
        self.memory = m

        for epoch in range(self.args.epochs):
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy_loss = 0
            for i, episode in enumerate(self.memory):
                actor_loss, critic_loss, entropy_loss = self.learn_episode(episode)
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy_loss
            loss = (total_actor_loss
                    + total_critic_loss * self.critic_update_weight
                    + total_entropy_loss * self.entropy_update_weight)
            loss /= len(self.memory)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.policy.critic.parameters(), 40)
            self.optimizer.step()

            self.summary_writer.add_scalar('totall_loss', loss.item(), global_step=ep + epoch)
            self.summary_writer.add_scalar('critic_loss', total_critic_loss.item()/len(self.memory), global_step=ep + epoch)
            self.summary_writer.add_scalar('actor_loss', total_actor_loss.item()/len(self.memory), global_step=ep + epoch)
            self.summary_writer.add_scalar('entropy_loss', total_entropy_loss.item()/len(self.memory), global_step=ep + epoch)
            self.summary_writer.add_scalar('critic_gradient_norm',
                                           sum([p.grad.data.norm(2) ** 2 for p in self.policy.critic.parameters()]) ** (
                                                       1. / 2), global_step=ep + epoch)
            self.summary_writer.add_scalar('actor_gradient_norm',
                                           sum([p.grad.data.norm(2) ** 2 for p in self.policy.actor.parameters()]) ** (
                                                       1. / 2), global_step=ep + epoch)
            self.summary_writer.add_histogram('action_var', self.policy.action_var, global_step=ep + epoch)

        self.policy_old.load_state_dict(self.policy.state_dict())

    def process_episode(self, e):
        lambdas = [self.args.lam ** i for i in range(len(e.rewards))]
        gammas = [self.args.gamma ** i for i in range(len(e.rewards))]
        lambdas = torch.FloatTensor(lambdas)
        gammas = torch.FloatTensor(gammas)
        coefs = gammas * lambdas
        coefs_tri = self.build_triangle_matrix(coefs).to(device)
        gammas_tri = self.build_triangle_matrix(gammas).to(device)

        e.coefs = coefs_tri

        e.rewards = torch.FloatTensor(e.rewards).to(device)
        returns = gammas_tri.matmul(e.rewards)
        e.returns = returns

        e.logprobs = torch.cat(e.logprobs).detach()
        e.actions = torch.cat(e.actions).detach()
        e.states = torch.cat(e.states).detach()
        return e

    def learn_episode(self, episode):

        state_values, new_actions = self.policy.evaluate(episode.states)
        logprobs = self.policy_sampler.get_logprobs(actions=new_actions, sampled_actions=episode.actions)
        entropy = self.policy_sampler.get_entropy(new_actions)

        next_state_values = torch.cat((state_values[1:], torch.zeros(1).to(device)))
        targets = episode.returns
        td_residuals = episode.rewards + next_state_values * self.args.gamma - state_values

        advantages = episode.coefs.matmul(td_residuals)
        # advantages = returns - state_values #+ next_state_value.squeeze() * terminal[-1]
        advantages = torch.norm(advantages)

        ratios = torch.exp(logprobs - episode.logprobs)

        surrogate = ratios * advantages
        surrogate_clipped = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages
        actor_loss = - (torch.min(surrogate, surrogate_clipped)).mean()
        critic_loss = (0.5 * td_residuals + 0.5 * self.loss(targets, state_values)).mean()
        entropy_loss = - entropy.mean()
        return actor_loss, critic_loss, entropy_loss

    def save(self, identity):
        path = Path(f'{self.args.model_dir}/{identity}')
        if path.exists():
            os.system(f'rm -rf {path}')
        path.mkdir(parents=True)
        torch.save(self.policy.state_dict(), f'{str(path)}/actor')


