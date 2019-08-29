from torch import nn
import torch
from ..BaseAgent import BaseAgent
from policies.ActorCriticPolicy import ActorCritic
import logging
import math


class Episode:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []


def build_triangle_matrix(v):
    tri = torch.zeros((len(v), len(v)))
    for i in range(len(v)):
        tri[i, i:] = v
        v = v[:-1]
    return tri


#TODO: dkl-penalty, GAE

class PPOAgent(BaseAgent):
    
    def add_arguments(self, parser):
        parser.add_argument('--dkl-penalty', help='Use penalty instead of surrogate objective clipping',
                            action='store_true')
        parser.add_argument('--lr', help='actor network learning rate', default=0.0001, type=float)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99, type=float)
        parser.add_argument('--batch-size', help='size of update memory', default=100, type=int)
        parser.add_argument('--epochs', help='epochs to update per step', default=5, type=int)
        parser.add_argument('--actor_hidden_units', help='Size of a hidden layers', default=[256, 128], nargs='+', type=int)
        parser.add_argument('--critic_hidden_units', help='Size of a hidden layers', default=[256, 128], nargs='+', type=int)
        parser.add_argument('--epsilon', help='clip objective threshold', default=0.2, type=float)
        parser.add_argument('--lam', help='Gae lambda parameter', default=0.5, type=float)
        parser.add_argument('--model-dir', help='directory for storing gym results', default='./saved_models')
        parser.add_argument('--cuda', help='Enable gpu optimization', action='store_true')
        return parser
    
    def __init__(self, observation_space, action_space, args_for_parse, summary_writer=None):
        super(PPOAgent, self).__init__(observation_space, action_space, args_for_parse, summary_writer)
        self.summary_writer = summary_writer

        if self.args.model_dir == 'polyaxon':
            from polyaxon_helper import get_outputs_path
            self.args.model_dir = get_outputs_path()

        self.gamma = self.args.gamma
        self.action_dim = action_space.shape[0]
        self.state_dim = observation_space.shape[0]
        self.critic_update_weight = 0.5
        self.entropy_update_weight = 0.01

        self.policy = ActorCritic(self.state_dim,
                                  self.policy_sampler.get_layer_size_before_sample(self.action_dim),
                                  self.args.actor_hidden_units, self.args.critic_hidden_units).to(self.args.device)
        self.policy_old = ActorCritic(self.state_dim,
                                      self.policy_sampler.get_layer_size_before_sample(self.action_dim),
                                      self.args.actor_hidden_units, self.args.critic_hidden_units).to(self.args.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr)
        self.loss = nn.MSELoss(reduce=False)

        self.step = 0
        self.memory = [Episode()]
        self.new_episode = False

        # Variables used for logging
        self.total_actor_loss = 0
        self.total_critic_loss = 0
        self.total_entropy_loss = 0

        self.last_logprobs = None

    def _act_normalized(self, state, episode):
        action = self.policy_old.act(state)
        action_sampled, logprob = self.add_variance(action)
        self.last_logprobs = logprob
        logging.debug('acting on {} with {}'.format(state, action))
        return action_sampled

    def _memorize_normalized(self, s, a, r, terminal, s_prim):
        self.memory[-1].rewards.append(r)
        self.memory[-1].states.append(s)
        # Sampling occurs from uniform distribution during random episodes
        logprobs = self.last_logprobs.to(self.args.device) if self.last_logprobs is not None \
                                         else torch.full((a.shape[0],), 1/2).to(self.args.device)
        # print('logprobs {}'.format(logprobs))
        self.memory[-1].logprobs.append(logprobs)
        self.memory[-1].actions.append(a)
        if terminal:
            self.new_episode = True

    def update(self, episode=0):
        if self.new_episode:
            if episode % self.args.batch_size == 0 and episode:
                self.memory = [self.process_episode(e) for e in self.memory]
                self.fit_batch(episode)
                self.memory = []
            self.memory.append(Episode())
            self.new_episode = False

    def fit_batch(self, ep):
        for epoch in range(self.args.epochs):
            self.total_actor_loss = 0
            self.total_critic_loss = 0
            self.total_entropy_loss = 0
            # logging.info('Memory length {}'.format(len(self.memory)))
            for i, episode in enumerate(self.memory):
                if episode.states.ndim == 0:
                    continue
                actor_loss, critic_loss, entropy_loss, last_variance = self.learn_episode(episode)
                self.total_actor_loss += actor_loss
                self.total_critic_loss += critic_loss
                self.total_entropy_loss += entropy_loss
            loss = (self.total_actor_loss
                    + self.total_critic_loss * self.critic_update_weight
                    + self.total_entropy_loss * self.entropy_update_weight)
            loss /= len(self.memory)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.policy.critic.parameters(), 40)
            self.optimizer.step()

            self.add_summary((ep//len(self.memory)) *self.args.epochs * len(self.memory) + epoch)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def add_summary(self, step):
        self.summary_writer.add_scalar('totall_loss', (self.total_actor_loss
                                                       + self.total_critic_loss * self.critic_update_weight
                                                       + self.total_entropy_loss * self.entropy_update_weight),
                                       global_step=step)
        self.summary_writer.add_scalar('critic_loss', self.total_critic_loss.item() / len(self.memory),
                                       global_step=step)
        self.summary_writer.add_scalar('actor_loss', self.total_actor_loss.item() / len(self.memory), global_step=step)
        self.summary_writer.add_scalar('entropy_loss', self.total_entropy_loss.item() / len(self.memory),
                                       global_step=step)
        self.summary_writer.add_scalar('critic_gradient_norm',
                                       sum([p.grad.data.norm(2) ** 2 for p in self.policy.critic.parameters()]) ** (
                                               1. / 2), global_step=step)
        self.summary_writer.add_scalar('actor_gradient_norm',
                                       sum([p.grad.data.norm(2) ** 2 for p in self.policy.actor.parameters()]) ** (
                                               1. / 2), global_step=step)


    def process_episode(self, e):
        lambdas = [self.args.lam ** i for i in range(len(e.rewards))]
        gammas = [self.args.gamma ** i for i in range(len(e.rewards))]
        lambdas = torch.FloatTensor(lambdas)
        gammas = torch.FloatTensor(gammas)
        coefs = gammas * lambdas
        coefs_tri = build_triangle_matrix(coefs).to(self.args.device)
        gammas_tri = build_triangle_matrix(gammas).to(self.args.device)

        e.coefs = coefs_tri

        e.rewards = torch.FloatTensor(e.rewards).to(self.args.device)
        logging.debug('rewards {}'.format(e.rewards.shape))
        logging.debug('gammas triangular {}'.format(gammas_tri))
        returns = gammas_tri.matmul(e.rewards)
        logging.debug('returns {}'.format(returns.shape))
        e.returns = returns

        e.logprobs = torch.cat(e.logprobs).detach()
        e.actions = torch.cat(e.actions).detach()
        e.states = torch.cat(e.states).detach()
        return e

    def learn_episode(self, episode):
        state_values, new_actions = self.policy.evaluate(episode.states)
        # logging.info('Learning state values {}, sampled vs new actions: {} vs {}'
        #                  .format(state_values, episode.actions, new_actions))
        logprobs, entropy = self.policy_sampler.get_logprobs(actions=new_actions, samples=episode.actions)
        last_variances = self.policy_sampler.get_variances(new_actions)

        logging.debug('returns size processed episodes {} '.format(episode.returns))
        episode.returns = episode.returns/torch.norm(episode.returns)
        next_state_values = torch.cat((state_values[1:], torch.zeros(1).to(self.args.device)))
        targets = episode.returns/torch.norm(episode.returns)
        # td_residuals = episode.rewards + next_state_values * self.args.gamma - state_values

        logging.debug('targets size processed episodes {} '.format(episode.returns))
        # advantages = episode.coefs.matmul(td_residuals)
        advantages = episode.returns - state_values.detach() #+ next_state_value.squeeze() * terminal[-1]
        # advantages = torch.norm(advantages)

        ratios = torch.exp(logprobs - episode.logprobs)

        surrogate = ratios * advantages
        surrogate_clipped = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages
        actor_loss = - (torch.min(surrogate, surrogate_clipped)).sum()
        # critic_loss = (0.5 * td_residuals + 0.5 * self.loss(targets, state_values)).mean()
        critic_loss = self.loss(targets.detach(), state_values).sum()

        # entropy_loss = - entropy.mean()
        entropy_loss = torch.zeros(1)
        return actor_loss, critic_loss, entropy_loss, last_variances

    def state_dict(self):
        return self.policy.state_dict()
