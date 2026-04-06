import argparse
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pettingzoo.butterfly import pistonball_v6

from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--shared_params", action="store_true")
args, _ = parser.parse_known_args()

ROLLOUTS = 512


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, state_space=state_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, self.num_actions), nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]) * 2.0, {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, state_space=state_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class SharedPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device,
                 agent_id, num_agents, shared_net, shared_log_std):
        Model.__init__(self, observation_space=observation_space, state_space=state_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.net = shared_net
        self.log_std_parameter = shared_log_std

    def compute(self, inputs, role):
        obs = inputs["observations"]
        one_hot = F.one_hot(
            torch.full((obs.shape[0],), self.agent_id, dtype=torch.long, device=obs.device),
            num_classes=self.num_agents,
        ).float()
        return self.net(torch.cat([obs, one_hot], dim=-1)) * 2.0, {"log_std": self.log_std_parameter}


class SharedValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device,
                 agent_id, num_agents, shared_net):
        Model.__init__(self, observation_space=observation_space, state_space=state_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.net = shared_net

    def compute(self, inputs, role):
        obs = inputs["observations"]
        one_hot = F.one_hot(
            torch.full((obs.shape[0],), self.agent_id, dtype=torch.long, device=obs.device),
            num_classes=self.num_agents,
        ).float()
        return self.net(torch.cat([obs, one_hot], dim=-1)), {}


class IPPOShared(IPPO):
    """IPPO with a single shared optimizer for all agents' shared weights."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        uid0 = self.possible_agents[0]
        shared_policy = self.policies[uid0]
        shared_value = self.values[uid0]
        if shared_policy is shared_value:
            params = shared_policy.parameters()
        else:
            params = itertools.chain(shared_policy.parameters(), shared_value.parameters())
        single_optimizer = torch.optim.Adam(params, lr=self.cfg.learning_rate[uid0][0])
        sched_cls = self.cfg.learning_rate_scheduler[uid0][0]
        sched_kw = self.cfg.learning_rate_scheduler_kwargs[uid0][0]
        single_scheduler = sched_cls(single_optimizer, **sched_kw) if sched_cls else None
        for uid in self.possible_agents:
            self.optimizers[uid] = single_optimizer
            self.schedulers[uid] = single_scheduler
            self.checkpoint_modules[uid]["optimizer"] = single_optimizer


set_seed(args.seed)

env = pistonball_v6.parallel_env(max_cycles=125, render_mode=None if args.headless else "human")
env = wrap_env(env)
device = env.device
num_agents = len(env.possible_agents)

if args.shared_params:
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).shape[0]
    shared_policy_net = nn.Sequential(
        nn.Linear(obs_dim + num_agents, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, act_dim), nn.Tanh(),
    ).to(device)
    shared_value_net = nn.Sequential(
        nn.Linear(obs_dim + num_agents, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    shared_log_std = nn.Parameter(torch.zeros(act_dim, device=device))
    models = {
        uid: {
            "policy": SharedPolicy(env.observation_space(uid), env.state_space(uid), env.action_space(uid), device,
                                   agent_id=i, num_agents=num_agents,
                                   shared_net=shared_policy_net, shared_log_std=shared_log_std),
            "value":  SharedValue(env.observation_space(uid), env.state_space(uid), env.action_space(uid), device,
                                  agent_id=i, num_agents=num_agents, shared_net=shared_value_net),
        }
        for i, uid in enumerate(env.possible_agents)
    }
else:
    models = {
        uid: {
            "policy": Policy(env.observation_space(uid), env.state_space(uid), env.action_space(uid), device),
            "value":  Value(env.observation_space(uid), env.state_space(uid), env.action_space(uid), device),
        }
        for uid in env.possible_agents
    }

memories = {
    uid: RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device)
    for uid in env.possible_agents
}

cfg = IPPO_CFG()
cfg.rollouts = ROLLOUTS
cfg.learning_epochs = 10
cfg.mini_batches = 8
cfg.discount_factor = 0.99
cfg.gae_lambda = 0.95
cfg.learning_rate = 3e-4
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
cfg.grad_norm_clip = 0.5
cfg.ratio_clip = 0.2
cfg.value_clip = 0.2
cfg.entropy_loss_scale = 0.01
cfg.value_loss_scale = 0.5
cfg.kl_threshold = 0
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space(env.possible_agents[0]), "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}

run_tag = "shared" if args.shared_params else "independent"
cfg.experiment.directory = "runs/torch/PistonBall"
cfg.experiment.experiment_name = f"IPPO_{run_tag}"
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0

AgentClass = IPPOShared if args.shared_params else IPPO
agent = AgentClass(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,
    cfg=cfg,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=device,
)

cfg_trainer = {"timesteps": 1_000_000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
