from copy import deepcopy
from typing import (
    Any,
    ClassVar,
    Dict,
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
from einops import rearrange
from gymnasium import spaces
from ot import sinkhorn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac.policies import (
    Actor,
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
    SACPolicy,
)
from torch.nn import functional as F

from imitation.algorithms import base as algo_base
from imitation.util import logger as imit_logger, networks, util

SelfSAC = TypeVar("SelfSAC", bound="OT")


class OT(algo_base.DemonstrationAlgorithm, SAC):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

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
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
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
        # OT
        reward_scale: float = 100.0,
        lfo: bool = False,
        num_expert_samples: int = 10,
        encoder_path: Optional[str] = None,
    ):
        SAC.__init__(
            self,
            policy,
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
        self.reward_scale = reward_scale
        self.lfo = lfo
        self.num_expert_samples = num_expert_samples

        if encoder_path is not None:
            if encoder_path == "dino":
                self.encoder = th.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitg14_reg"
                )
            else:
                try:
                    algo = SAC.load(encoder_path, device=device)
                    self.encoder = algo.policy.actor.features_extractor
                except AttributeError:
                    algo = PPO.load(encoder_path, device=device)
                    self.encoder = algo.policy.features_extractor
        else:
            self.encoder = None

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        self.demonstrations = demonstrations

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        actions_history = []
        obs_history = []
        next_obs_history = []
        rewards_history = []
        dones_history = []
        infos_history = []
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Avoid modification by reference
            next_obs = deepcopy(new_obs)
            # As the VecEnv resets automatically, new_obs is already the
            # first observation of the next episode
            for i, done in enumerate(dones):
                if done and infos[i].get("terminal_observation") is not None:
                    if isinstance(next_obs, dict):
                        next_obs_ = infos[i]["terminal_observation"]
                        # VecNormalize normalizes the terminal observation
                        if self._vec_normalize_env is not None:
                            next_obs_ = self._vec_normalize_env.unnormalize_obs(
                                next_obs_
                            )
                        # Replace next obs for the correct envs
                        for key in next_obs.keys():
                            next_obs[key][i] = next_obs_[key]
                    else:
                        next_obs[i] = infos[i]["terminal_observation"]
                        # VecNormalize normalizes the terminal observation
                        if self._vec_normalize_env is not None:
                            next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                                next_obs[i, :]
                            )

            actions_history.append(buffer_actions)
            obs_history.append(self._last_obs)
            next_obs_history.append(next_obs)
            rewards_history.append(rewards)
            dones_history.append(dones)
            infos_history.append(infos)
            self._last_obs = next_obs

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        callback.on_rollout_end()

        actions_history = np.stack(actions_history, axis=1)
        obs_history = np.stack(obs_history, axis=1)
        next_obs_history = np.stack(next_obs_history, axis=1)
        rewards_history = np.stack(rewards_history, axis=1)
        dones_history = np.stack(dones_history, axis=1)
        infos_history = np.stack(infos_history, axis=1)

        # Calculate rewards
        new_rewards = self._calc_reward(actions_history, obs_history, next_obs_history)
        rewards_history = new_rewards
        return_mean = np.sum(rewards_history, axis=1).mean()
        self.logger.record("rollout/return_mean", return_mean)

        # Add data to replay buffer
        for i in range(obs_history.shape[1]):
            self.replay_buffer.add(
                obs_history[:, i],
                next_obs_history[:, i],
                actions_history[:, i],
                rewards_history[:, i],
                dones_history[:, i],
                infos_history[:, i],
            )
        return RolloutReturn(
            num_collected_steps,
            num_collected_episodes,
            continue_training,
        )

    def _sample_trajs(self, n_trajs: int):
        if self.demonstrations is None:
            raise ValueError("No demonstrations provided.")
        if len(self.demonstrations) < n_trajs or n_trajs < 0:
            n_trajs = len(self.demonstrations)
        idxs = np.random.choice(len(self.demonstrations), n_trajs)
        return [self.demonstrations[idx] for idx in idxs]

    def _cosine_distance(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        x = util.safe_to_tensor(x).to(th.float32).to(self.device)
        y = util.safe_to_tensor(y).to(th.float32).to(self.device)
        cost_matrix = th.mm(x, y.T)
        x_norm = th.norm(x, p=2, dim=-1)
        y_norm = th.norm(y, p=2, dim=-1)
        x_n = x_norm.unsqueeze(1)
        y_n = y_norm.unsqueeze(1)
        norms = th.mm(x_n, y_n.T)
        cost_matrix = 1 - cost_matrix / norms
        return cost_matrix

    def _optimal_transport_plan(
        self, X, Y, cost_matrix, niter=1000, epsilon=0.01, method="sinkhorn_log"
    ):
        X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
        Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
        c_m = cost_matrix.data.detach().cpu().numpy()
        c_m = c_m / c_m.max()
        transport_plan = sinkhorn(
            X_pot, Y_pot, c_m, epsilon, numItermax=niter, method=method
        )
        transport_plan = th.from_numpy(transport_plan).to(th.float32).to(X.device)
        transport_plan.requires_grad = False
        return transport_plan

    def _calc_cost(self, agent_inputs, expert_inputs):
        cost_matrix = self._cosine_distance(agent_inputs, expert_inputs)
        transport_plan = self._optimal_transport_plan(
            agent_inputs, expert_inputs, cost_matrix
        )
        costs = th.diag(th.mm(transport_plan, cost_matrix.T))
        return costs

    def _calc_reward(self, actions_history, obs_history, next_obs_history):
        actions_history = (
            util.safe_to_tensor(actions_history).to(th.float32).to(self.device)
        )
        obs_history = util.safe_to_tensor(obs_history).to(th.float32).to(self.device)
        next_obs_history = (
            util.safe_to_tensor(next_obs_history).to(th.float32).to(self.device)
        )

        if self.encoder is not None:
            b = obs_history.shape[0]
            t = obs_history.shape[1]
            obs_history = rearrange(obs_history, "b t c h w -> (b t) c h w")
            next_obs_history = rearrange(next_obs_history, "b t c h w -> (b t) c h w")
            with th.no_grad():
                obs_history = self.encoder(obs_history)
                next_obs_history = self.encoder(next_obs_history)
            obs_history = rearrange(obs_history, "(b t) f -> b t f", b=b, t=t)
            next_obs_history = rearrange(next_obs_history, "(b t) f -> b t f", b=b, t=t)

        agent_inputs = [obs_history]
        if self.lfo:
            agent_inputs.append(next_obs_history)
        else:
            agent_inputs.append(actions_history)
        agent_inputs = th.cat(agent_inputs, dim=-1)

        new_rewards = []
        for agent_input in agent_inputs:
            sampled_trajs = self._sample_trajs(self.num_expert_samples)
            distance_list = []
            rewards_list = []
            for traj in sampled_trajs:
                expert_curr_obs = traj.obs[:-1]
                expert_next_obs = traj.obs[1:]
                expert_curr_obs = (
                    util.safe_to_tensor(expert_curr_obs).to(th.float32).to(self.device)
                )
                expert_next_obs = (
                    util.safe_to_tensor(expert_next_obs).to(th.float32).to(self.device)
                )

                if self.encoder is not None:
                    with th.no_grad():
                        expert_curr_obs = self.encoder(expert_curr_obs)
                        expert_next_obs = self.encoder(expert_next_obs)

                expert_inputs = [expert_curr_obs]
                if self.lfo:
                    expert_inputs.append(expert_next_obs)
                else:
                    expert_inputs.append(traj.acts)
                expert_inputs = th.cat(expert_inputs, dim=-1)

                costs = (
                    self._calc_cost(agent_input, expert_inputs).detach().cpu().numpy()
                )
                rewards = -self.reward_scale * costs

                distance_list.append(np.sum(costs))
                rewards_list.append(rewards)
            closest_demo_index = np.argmin(distance_list)
            rewards = rewards_list[closest_demo_index]
            new_rewards.append(rewards)
        new_rewards = np.stack(new_rewards, axis=0)
        return new_rewards
