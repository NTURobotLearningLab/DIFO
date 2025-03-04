import dataclasses
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update, update_learning_rate
from stable_baselines3.sac.policies import (
    Actor,
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
    SACPolicy,
)
from torch.nn import functional as F

from imitation.algorithms import base as algo_base
from imitation.data import buffer, rollout, types, wrappers
from imitation.util import logger as imit_logger, networks, util

SelfIQLearn = TypeVar("SelfIQLearn", bound="IQLearn")


class IQLearn(algo_base.DemonstrationAlgorithm, SAC):
    """
    Inverse Q-Learning (IQ-Learn) algorithm.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        *,
        learning_rate: Union[float, Schedule] = 3e-4,
        actor_learning_rate: Optional[Union[float, Schedule]] = None,
        critic_learning_rate: Optional[Union[float, Schedule]] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # DemonstrationAlgorithm
        demonstrations: Optional[algo_base.AnyTransitions] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        # IQ-Learn
        lfo: bool = False,
        div_method: str = "kl",  # kl, kl2, kl_fix, hellinger, js
        loss_method: str = "value",  # value_expert, value, v0
        grad_pen: bool = False,
        lambda_gp: float = 10.0,
        chi: bool = False,
        regularize: bool = False,
        use_target_value: bool = False,
    ):
        SAC.__init__(
            self,
            SACPolicy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            _init_setup_model=_init_setup_model,
        )
        algo_base.DemonstrationAlgorithm.__init__(
            self,
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        self.lfo = lfo
        self.div_method = div_method
        self.loss_method = loss_method
        self.grad_pen = grad_pen
        self.lambda_gp = lambda_gp
        self.chi = chi
        self.regularize = regularize
        self.use_target_value = use_target_value

        actor_learning_rate = actor_learning_rate or learning_rate
        critic_learning_rate = critic_learning_rate or learning_rate
        update_learning_rate(self.critic.optimizer, critic_learning_rate)
        update_learning_rate(self.actor.optimizer, actor_learning_rate)
        if self.ent_coef_optimizer is not None:
            update_learning_rate(self.ent_coef_optimizer, self.learning_rate)

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        self._demo_data_loader = algo_base.make_data_loader(
            demonstrations,
            self.batch_size,
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def _next_expert_batch(self) -> Mapping:
        assert self._endless_expert_iterator is not None
        return next(self._endless_expert_iterator)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        loss_dicts = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Compute critic loss
            batch = self._make_disc_train_batches(
                gen_samples=replay_data, expert_samples=None
            )
            critic_loss, loss_dict = self.critic_loss(batch)
            loss_dicts.append(loss_dict)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        for loss_name in loss_dicts[0].keys():
            self.logger.record(
                f"train/iq_{loss_name}",
                np.mean([loss_dict[loss_name] for loss_dict in loss_dicts]),
            )
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    @property
    def curr_ent_coef(self) -> th.Tensor:
        """Return the entropy coefficient (temperature) as a tensor"""
        if self.ent_coef == "auto":
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            return th.exp(self.log_ent_coef.detach())
        else:
            return self.ent_coef_tensor

    def getV(self, obs: th.Tensor, use_target=False) -> th.Tensor:
        """Compute the value function V(s) for a given state s"""
        actions_pi, log_prob = self.actor.action_log_prob(obs)
        if use_target:
            with th.no_grad():
                q_values_pi = th.cat(self.critic_target(obs, actions_pi), dim=1)
        else:
            q_values_pi = th.cat(self.critic(obs, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        values = min_qf_pi - self.curr_ent_coef * log_prob.reshape(-1, 1)
        return values

    def critic_loss(self, batch) -> Tuple[th.Tensor, Dict[str, float]]:
        """Compute the critic loss for a given batch of observations"""
        obs = batch["state"]
        next_obs = batch["next_state"]
        action = batch["action"]

        current_v = self.getV(obs)
        next_v = self.getV(next_obs, use_target=self.use_target_value)

        Q_losses = []
        loss_dicts = []
        current_Qs = self.critic(obs, action)
        for current_Q in current_Qs:
            Q_loss, loss_dict = self.iq_loss(current_Q, current_v, next_v, batch)
            Q_losses.append(Q_loss)
            loss_dicts.append(loss_dict)
        critic_loss = th.stack(Q_losses).mean()

        loss_dict = {}
        for k in loss_dicts[0].keys():
            loss_dict[k] = np.mean([loss_dict[k] for loss_dict in loss_dicts])

        return critic_loss, loss_dict

    def iq_loss(self, current_Q, current_v, next_v, batch):
        """
        Full IQ-Learn objective with other divergences and options.

        Copyright 2022 Div Garg. All rights reserved.

        Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
        """

        obs = batch["state"]
        action = batch["action"]
        done = batch["done"].reshape(-1, 1)
        is_expert = batch["labels_expert_is_one"].reshape(-1, 1)

        loss_dict = {}
        # keep track of value of initial states
        v0 = self.getV(obs[is_expert.squeeze(1), ...]).mean()
        loss_dict["v0"] = v0.item()

        #  calculate 1st term for IQ los
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        y = (1 - done) * self.gamma * next_v
        reward = (current_Q - y)[is_expert]

        with th.no_grad():
            # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
            if self.div_method == "hellinger":
                phi_grad = 1 / (1 + reward) ** 2
            elif self.div_method == "kl":
                # original dual form for kl divergence (sub optimal)
                phi_grad = th.exp(-reward - 1)
            elif self.div_method == "kl2":
                # biased dual form for kl divergence
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif self.div_method == "kl_fix":
                # our proposed unbiased form for fixing kl divergence
                phi_grad = th.exp(-reward)
            elif self.div_method == "js":
                # jensen–shannon
                phi_grad = th.exp(-reward) / (2 - th.exp(-reward))
            else:
                phi_grad = 1
        loss = -(phi_grad * reward).mean()
        loss_dict["softq_loss"] = loss.item()

        # calculate 2nd term for IQ loss, we show different sampling strategies
        if self.loss_method == "value_expert":
            # sample using only expert states (works offline)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (current_v - y)[is_expert].mean()
            loss += value_loss
            loss_dict["value_loss"] = value_loss.item()

        elif self.loss_method == "value":
            # sample using expert and policy states (works online)
            # E_(ρ)[V(s) - γV(s')]
            value_loss = (current_v - y).mean()
            loss += value_loss
            loss_dict["value_loss"] = value_loss.item()

        elif self.loss_method == "v0":
            # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` strategy)
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            loss_dict["v0_loss"] = v0_loss.item()
        else:
            raise ValueError(
                f"This loss method is not supported: {self.loss_method}. Please choose from ['value_expert', 'value', 'v0']"
            )

        if self.grad_pen:
            # add a gradient penalty to loss (Wasserstein_1 metric)
            raise NotImplementedError("Gradient penalty is not implemented yet")
            gp_loss = agent.critic_net.grad_pen(
                obs[is_expert.squeeze(1), ...],
                action[is_expert.squeeze(1), ...],
                obs[~is_expert.squeeze(1), ...],
                action[~is_expert.squeeze(1), ...],
                self.lambda_gp,
            )
            loss_dict["gp_loss"] = gp_loss.item()
            loss += gp_loss

        if self.chi:  # TODO: Deprecate method.chi argument for method.div
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
            y = (1 - done) * self.gamma * next_v

            reward = current_Q - y
            chi2_loss = 1 / (4 * self.curr_ent_coef) * (reward**2)[is_expert].mean()
            loss += chi2_loss
            loss_dict["chi2_loss"] = chi2_loss.item()

        if self.regularize:
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
            y = (1 - done) * self.gamma * next_v

            reward = current_Q - y
            chi2_loss = 1 / (4 * self.curr_ent_coef) * (reward**2).mean()
            loss += chi2_loss
            loss_dict["regularize_loss"] = chi2_loss.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Iterator[Mapping[str, th.Tensor]]:
        """Build and return training minibatches for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Yields:
            The training minibatch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """
        batch_size = self.batch_size

        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self.replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )
            gen_samples_dataclass = self.replay_buffer.sample(batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)

        gen_samples = dict(
            obs=gen_samples.observations,
            acts=gen_samples.actions,
            next_obs=gen_samples.next_observations,
            dones=gen_samples.dones.flatten(),
        )

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need to have exactly `demo_batch_size` number of expert and "
                "generator samples, each. "
                f"(n_gen={len(gen_samples['obs'])} "
                f"n_expert={len(expert_samples['obs'])} "
                f"demo_batch_size={batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to Tensors.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], np.ndarray):
                    d[k] = th.from_numpy(d[k])
                d[k] = d[k].to(self.device)
        assert isinstance(gen_samples["obs"], th.Tensor)
        assert isinstance(expert_samples["obs"], th.Tensor)

        # Check dimensions.
        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])
        assert batch_size == len(gen_samples["acts"])
        assert batch_size == len(gen_samples["next_obs"])

        start = 0
        end = start + batch_size
        # take minibatch slice (this creates views so no memory issues)
        expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
        gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

        if self.lfo:
            expert_batch["acts"] = gen_batch["acts"]

        # Concatenate rollouts, and label each row as expert or generator.
        obs = th.cat([expert_batch["obs"], gen_batch["obs"]])
        acts = th.cat([expert_batch["acts"], gen_batch["acts"]])
        next_obs = th.cat([expert_batch["next_obs"], gen_batch["next_obs"]])
        dones = th.cat([expert_batch["dones"], gen_batch["dones"]])
        # notice that the labels use the convention that expert samples are
        # labelled with 1 and generator samples with 0.
        labels_expert_is_one = th.cat(
            [
                th.ones(batch_size, dtype=th.bool),
                th.zeros(batch_size, dtype=th.bool),
            ],
        ).to(self.device)
        batch_dict = {
            "state": obs,
            "action": acts,
            "next_state": next_obs,
            "done": dones,
            "labels_expert_is_one": labels_expert_is_one,
            "log_policy_act_prob": None,
        }
        return batch_dict
