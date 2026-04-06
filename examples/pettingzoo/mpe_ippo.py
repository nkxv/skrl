import argparse
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pettingzoo.mpe import simple_reference_v3, simple_speaker_listener_v4, simple_spread_v3

from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="spread",
    choices=["spread", "reference", "speaker_listener"],
    help="MPE environment to train on",
)
parser.add_argument("--n_agents", type=int, default=3, help="Number of agents (spread only)")
parser.add_argument("--max_cycles", type=int, default=25, help="Maximum steps per episode")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
parser.add_argument("--shared_params", action="store_true", help="Use shared model parameters across agents")
args, _ = parser.parse_known_args()

# speaker_listener has heterogeneous agents (different obs/action sizes) — sharing weights is not possible
if args.shared_params and args.env == "speaker_listener":
    logger.error("--shared_params is not supported for 'speaker_listener': speaker and listener have different "
                 "observation and action space sizes and cannot share a backbone network.")
    exit(1)


# single source of truth — memory_size AND cfg.rollouts must be equal
ROLLOUTS = 128


# ---- Non-shared models -------------------------------------------------------

class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=unnormalized_log_prob)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# ---- Shared-parameter models (homogeneous envs only) -------------------------

class SharedPolicy(CategoricalMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device,
                 agent_id: int, num_agents: int, shared_net, unnormalized_log_prob=True):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=unnormalized_log_prob)

        self.agent_id = agent_id
        self.num_agents = num_agents
        # shared_net input: num_observations + num_agents (one-hot agent ID appended in compute)
        self.net = shared_net

    def compute(self, inputs, role):
        obs = inputs["observations"]
        one_hot = F.one_hot(
            torch.full((obs.shape[0],), self.agent_id, dtype=torch.long, device=obs.device),
            num_classes=self.num_agents,
        ).float()
        return self.net(torch.cat([obs, one_hot], dim=-1)), {}


class SharedValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device,
                 agent_id: int, num_agents: int, shared_net):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
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


# ---- IPPOShared subclass -----------------------------------------------------

class IPPOShared(IPPO):
    """IPPO variant with parameter sharing.

    IPPO.__init__ creates one Adam optimizer per agent UID. When all agents share
    the same model weights, this produces N optimizers with independent moment
    estimates (m, v) all tracking the same parameters — the Adam updates become
    inconsistent after the first step.

    This subclass replaces the N optimizers with a single shared optimizer after
    the parent __init__ runs, ensuring one consistent set of Adam moments for the
    shared weights.
    """

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


# ---- Environment setup -------------------------------------------------------

set_seed(args.seed)

render_mode = None if args.headless else "human"

if args.env == "spread":
    raw_env = simple_spread_v3.parallel_env(N=args.n_agents, max_cycles=args.max_cycles, render_mode=render_mode)
elif args.env == "reference":
    raw_env = simple_reference_v3.parallel_env(max_cycles=args.max_cycles, render_mode=render_mode)
elif args.env == "speaker_listener":
    raw_env = simple_speaker_listener_v4.parallel_env(max_cycles=args.max_cycles, render_mode=render_mode)

env = wrap_env(raw_env)
device = env.device
num_agents = len(env.possible_agents)


# ---- Model and memory construction -------------------------------------------

if args.shared_params:
    # Homogeneous envs only (speaker_listener is rejected above).
    # All agents share the same obs/action space, so we read from agent 0.
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n  # Discrete.n

    shared_policy_net = nn.Sequential(
        nn.Linear(obs_dim + num_agents, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, act_dim),
    ).to(device)
    shared_value_net = nn.Sequential(
        nn.Linear(obs_dim + num_agents, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    models = {
        uid: {
            "policy": SharedPolicy(
                env.observation_space(uid), env.state_space(uid), env.action_space(uid), device,
                agent_id=i, num_agents=num_agents, shared_net=shared_policy_net,
            ),
            "value": SharedValue(
                env.observation_space(uid), env.state_space(uid), env.action_space(uid), device,
                agent_id=i, num_agents=num_agents, shared_net=shared_value_net,
            ),
        }
        for i, uid in enumerate(env.possible_agents)
    }
else:
    models = {
        uid: {
            "policy": Policy(env.observation_space(uid), env.state_space(uid), env.action_space(uid), device),
            "value": Value(env.observation_space(uid), env.state_space(uid), env.action_space(uid), device),
        }
        for uid in env.possible_agents
    }

# memory_size == ROLLOUTS == cfg.rollouts — single constant
memories = {
    uid: RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device)
    for uid in env.possible_agents
}


# ---- IPPO configuration ------------------------------------------------------

cfg = IPPO_CFG()
cfg.rollouts = ROLLOUTS
cfg.learning_epochs = 10
cfg.mini_batches = 4
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

# Per-agent preprocessor kwargs handle heterogeneous obs spaces (e.g. speaker_listener).
# RunningStandardScaler always uses the raw obs space — never the augmented one.
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {
    uid: {"size": env.observation_space(uid), "device": device}
    for uid in env.possible_agents
}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}

run_tag = "shared" if args.shared_params else "independent"
cfg.experiment.directory = f"runs/torch/MPE_{args.env}"
cfg.experiment.experiment_name = f"IPPO_{run_tag}"
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0


# ---- Agent, trainer, run -----------------------------------------------------

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
