import dataclasses
import os
import os.path as osp
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3.common import policies, torch_layers, utils, vec_env
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm, trange

from imitation.algorithms import base as algo_base
from imitation.algorithms.bc import (
    BC,
    BatchIteratorWithEpochEndCallback,
    enumerate_batches,
    reconstruct_policy,
)
from imitation.data import buffer, rollout, types
from imitation.policies import base as policy_base
from imitation.util import logger as imit_logger, util


class InvFunc(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=64):
        super().__init__()
        self.is_img = len(obs_shape) == 3

        if self.is_img:
            n = obs_shape[1]
            m = obs_shape[2]
            image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
            self.head = nn.Sequential(
                nn.Conv2d(2 * obs_shape[0], 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(image_embedding_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0]),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(2 * obs_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0]),
            )

    def forward(self, state_1, state_2):
        if self.is_img:
            x = th.cat([state_1, state_2], dim=1)
        else:
            x = th.cat([state_1, state_2], dim=-1)
        return self.head(x)


@dataclasses.dataclass(frozen=True)
class BCOTrainingMetrics:
    """Container for the different components of behavior cloning loss."""

    neglogp: th.Tensor
    entropy: Optional[th.Tensor]
    ent_loss: th.Tensor  # set to 0 if entropy is None
    prob_true_act: th.Tensor
    l2_norm: th.Tensor
    l2_loss: th.Tensor
    loss: th.Tensor


@dataclasses.dataclass(frozen=True)
class BCOLossCalculator:
    """Functor to compute the loss used in Behavior Cloning."""

    ent_weight: float
    l2_weight: float

    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, th.Tensor],
        ],
        acts: Union[th.Tensor, np.ndarray],
    ) -> BCOTrainingMetrics:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        tensor_obs = types.map_maybe_dict(
            util.safe_to_tensor,
            types.maybe_unwrap_dictobs(obs),
        )
        acts = util.safe_to_tensor(acts)

        # policy.evaluate_actions's type signatures are incorrect.
        # See https://github.com/DLR-RM/stable-baselines3/issues/1679
        (_, log_prob, entropy) = policy.evaluate_actions(
            tensor_obs,  # type: ignore[arg-type]
            acts,
        )
        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [th.sum(th.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, th.Tensor)

        ent_loss = -self.ent_weight * (entropy if entropy is not None else th.zeros(1))
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        return BCOTrainingMetrics(
            neglogp=neglogp,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            loss=loss,
        )


class BCOLogger:
    """Utility class to help logging information relevant to BCO."""

    def __init__(self, logger: imit_logger.HierarchicalLogger):
        """Create new BCO logger.
        Args:
            logger: The logger to feed all the information to.
        """
        self._logger = logger
        self._tensorboard_step = 0
        self._current_epoch = 0

    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0

    def log_epoch(self, epoch_number):
        self._current_epoch = epoch_number

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        num_samples_so_far: int,
        training_metrics: BCOTrainingMetrics,
        rollout_stats: Mapping[str, float],
    ):
        self._logger.record("bc/epoch", self._current_epoch)
        self._logger.record("bc/batch", batch_num)
        self._logger.record("bc/samples_so_far", num_samples_so_far)
        for k, v in training_metrics.__dict__.items():
            self._logger.record(f"bc/{k}", float(v) if v is not None else None)

        for k, v in rollout_stats.items():
            if "return" in k and "monitor" not in k:
                self._logger.record("rollout/" + k, v)
            if "success" in k:
                self._logger.record("rollout/" + k, v)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state


@dataclasses.dataclass(frozen=True)
class RolloutStatsComputer:
    """Computes statistics about rollouts.

    Args:
        venv: The vectorized environment in which to compute the rollouts.
        n_episodes: The number of episodes to base the statistics on.
    """

    venv: Optional[vec_env.VecEnv]
    n_episodes: int

    # TODO(shwang): Maybe instead use a callback that can be shared between
    #   all algorithms' `.train()` for generating rollout stats.
    #   EvalCallback could be a good fit:
    #   https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback
    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        rng: np.random.Generator,
    ) -> Mapping[str, float]:
        if self.venv is not None and self.n_episodes > 0:
            trajs = rollout.generate_trajectories(
                policy,
                self.venv,
                rollout.make_min_episodes(self.n_episodes),
                rng=rng,
            )
            return rollout.rollout_stats(trajs)
        else:
            return dict()


def find_dir(dir_name: str) -> Path:
    if dir_name is None:
        return None
    if isinstance(dir_name, list):
        return None
    target = Path(dir_name).parent
    if osp.isfile(dir_name):
        target = target.parent
    try:
        os.makedirs(target, exist_ok=True)
        return Path(dir_name)
    except OSError:
        raise FileNotFoundError(f"Could not create directory {dir_name}")


def reparam_sample(dist):
    """
    A general method for updating either a categorical or normal distribution.
    In the case of a Categorical distribution, the logits are just returned
    """
    if isinstance(dist, th.distributions.Normal):
        return dist.rsample()
    elif isinstance(dist, th.distributions.Categorical):
        return dist.logits
    else:
        raise ValueError("Unrecognized distribution")


def compute_ac_loss(pred_actions, true_actions, ac_space):
    if isinstance(pred_actions, th.distributions.Distribution):
        pred_actions = reparam_sample(pred_actions)

    if isinstance(ac_space, spaces.Discrete):
        loss = F.cross_entropy(pred_actions, true_actions.view(-1).long())
    else:
        loss = F.mse_loss(pred_actions, true_actions)
    return loss


class BCO(BC):
    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        policy: Optional[policies.ActorCriticPolicy] = None,
        bco_inv_lr: float = 0.0001,  # The learning rate of the action inverse model.
        bco_inv_batch_size: int = 32,  # The batch size for the inverse action model training.
        transition_buffer_size: int = 1_000_000,  # The size of the replay buffer.
        inv_hidden_dim: int = 64,  # The hidden dimension of the inverse model.
        save_dir: types.AnyPath = None,  # Directory to save the model to.
        demonstrations: Optional[algo_base.AnyTransitions] = None,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 2e-3,
        device: Union[str, th.device] = "cuda",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """
        Initialize the BCO (Behavioral Cloning from Observations) algorithm.

        Args:
            venv (vec_env.VecEnv): The vectorized environment.
            observation_space (gym.Space): The observation space of the environment.
            action_space (gym.Space): The action space of the environment.
            rng (np.random.Generator): The random number generator.
            policy (Optional[policies.ActorCriticPolicy]): The policy network used for training.
            bco_alpha_T (int): Number of online updates.
            bco_alpha (float): Size of each online update.
            bco_inv_lr (float): The learning rate of the action inverse model.
            bco_inv_epochs (int): The number of epochs when training the inverse model.
            bco_inv_batch_size (int): The batch size for the inverse action model training.
            bco_inv_eval_holdout (float): The fraction of data that should be withheld when training the inverse model and later used for evaluation.
            bco_inv_load (Optional[str]): If specified, the inverse model will be loaded from here and not trained on the random exploration phase. However, it **will** be trained on all subsequent alpha updates.
            bco_expl_steps (int): Number of random exploration steps.
            bco_expl_load (Optional[str]): If specified, the random exploration data will be loaded from here.
            save_dir (types.AnyPath): Directory to save the model to.
            demonstrations (Optional[algo_base.AnyTransitions]): Demonstrations data for behavioral cloning.
            batch_size (int): The batch size for training.
            minibatch_size (Optional[int]): The size of each minibatch for training.
            optimizer_cls (Type[torch.optim.Optimizer]): The optimizer class to use.
            optimizer_kwargs (Optional[Mapping[str, Any]]): Additional keyword arguments for the optimizer.
            ent_weight (float): The weight for the entropy loss term.
            l2_weight (float): The weight for the L2 regularization term.
            device (Union[str, torch.device]): The device to use for training.
            num_processes (int): The number of parallel processes.
            custom_logger (Optional[imit_logger.HierarchicalLogger]): Custom logger for logging.

        """
        self.venv = venv

        self.demonstration_size = np.sum(
            [len(traj) for traj in demonstrations]
        )  # |I pre| in the paper

        self.transition_buffer = buffer.ReplayBuffer(transition_buffer_size, venv)

        self.bco_inv_lr = bco_inv_lr
        self.bco_inv_batch_size = bco_inv_batch_size

        self._env_steps = 0
        self.lr_updates = None
        self.device = device

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            rng=rng,
            policy=policy,
            demonstrations=demonstrations,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            ent_weight=ent_weight,
            l2_weight=l2_weight,
            device=device,
            custom_logger=custom_logger,
        )

        self._bco_logger = BCOLogger(self.logger)
        del self._bc_logger
        if save_dir is not None:
            try:
                self.save_dir = find_dir(save_dir) / "models"
            except FileNotFoundError as e:
                self.logger.record(f"Could not create directory {save_dir}.")
                self.logger.record(f"Error: {e}")
                self.save_dir = None

        self.inv_func = InvFunc(
            observation_space.shape,
            action_space.shape,
            hidden_dim=inv_hidden_dim,
        )
        print(self.inv_func)
        self.inv_func = self.inv_func.to(self.device)
        self.inv_opt = optim.Adam(self.inv_func.parameters(), lr=self.bco_inv_lr)

        self.loss_calculator = BCOLossCalculator(ent_weight, l2_weight)

        assert self._demo_data_loader is not None

    def _train_inv_func(self, bco_inv_steps: int) -> List[float]:
        infer_ac_losses = []
        for i in trange(bco_inv_steps, dynamic_ncols=True):
            batch = self.transition_buffer.sample(self.bco_inv_batch_size)

            obs = util.safe_to_tensor(batch.obs).to(self.device).to(th.float32)
            next_obs = (
                util.safe_to_tensor(batch.next_obs).to(self.device).to(th.float32)
            )
            actions = util.safe_to_tensor(batch.acts).to(self.device).to(th.float32)

            pred_action = self.inv_func(obs, next_obs)
            loss = compute_ac_loss(
                pred_action,
                actions,
                self.policy.action_space,
            )
            infer_ac_losses.append(loss.item())

            self.inv_opt.zero_grad()
            loss.backward()
            self.inv_opt.step()

        return infer_ac_losses

    def save_model(self, model, save_name: str):
        if self.save_dir is None:
            return
        save_path = self.save_dir / save_name
        save_path = find_dir(save_path)
        th.save(model.state_dict(), save_path)

    def _train_loader(
        self,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        """
        yield the batches with stats from the demonstration data loader (expert data)
        """
        if reset_tensorboard:
            self._bco_logger.reset_tensorboard_steps()
        self._bco_logger.log_epoch(0)

        def _on_epoch_end(epoch_number: int):
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(
                    f"Epoch {epoch_number} {total_num_epochs_str}",
                    pos=1,
                )
            self._bco_logger.log_epoch(epoch_number + 1)
            if on_epoch_end is not None:
                on_epoch_end()

        mini_per_batch = self.batch_size // self.minibatch_size
        n_minibatches = n_batches * mini_per_batch if n_batches is not None else None
        assert self._demo_data_loader is not None
        demonstration_batches = BatchIteratorWithEpochEndCallback(
            self._demo_data_loader,
            n_epochs,
            n_minibatches,
            _on_epoch_end,
        )
        batches_with_stats = enumerate_batches(demonstration_batches)
        tqdm_progress_bar: Optional[tqdm] = None

        if progress_bar:
            batches_with_stats = tqdm(
                batches_with_stats,
                unit="batch",
                total=n_minibatches,
                dynamic_ncols=True,
            )
            tqdm_progress_bar = batches_with_stats
        return batches_with_stats

    @property
    def policy(self) -> policies.ActorCriticPolicy:
        return self._policy

    def train_bc(
        self,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        batches_loader: Iterable[Tuple[Tuple[int, int, int], Dict[str, np.ndarray]]]
        batches_loader = self._train_loader(
            n_epochs=n_epochs,
            n_batches=n_batches,
            on_epoch_end=on_epoch_end,
            progress_bar=progress_bar,
            reset_tensorboard=reset_tensorboard,
        )

        compute_rollout_stats = RolloutStatsComputer(
            log_rollouts_venv,
            log_rollouts_n_episodes,
        )

        def process_batch():
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch_num % log_interval == 0:
                rollout_stats = compute_rollout_stats(self.policy, self.rng)
                self._bco_logger.log_batch(
                    batch_num,
                    minibatch_size,
                    num_samples_so_far,
                    training_metrics,
                    rollout_stats,
                )
            if on_batch_end is not None:
                on_batch_end()

        self.optimizer.zero_grad()
        for (
            batch_num,
            minibatch_size,
            num_samples_so_far,
        ), batch in batches_loader:
            obs_tensor: Union[th.Tensor, Dict[str, th.Tensor]]
            next_obs_tensor: Union[th.Tensor, Dict[str, th.Tensor]]
            # unwraps the observation if it's a dictobs and converts arrays to tensors
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["obs"]),
            )
            next_obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["next_obs"]),
            )

            s0 = obs_tensor.to(self.device).float()
            s1 = next_obs_tensor.to(self.device).float()
            # Perform inference on the expert states
            with th.no_grad():
                pred_actions = self.inv_func(s0, s1).to(self.device)

            training_metrics = self.loss_calculator(
                self.policy, obs_tensor, pred_actions
            )

            # Renormalise the loss to be averaged over the whole
            # batch size instead of the minibatch size.
            # If there is an incomplete batch, its gradients will be
            # smaller, which may be helpful for stability.
            loss = training_metrics.loss * minibatch_size / self.batch_size
            loss.backward()
            batch_num = batch_num * self.minibatch_size // self.batch_size
            if num_samples_so_far % self.batch_size == 0:
                process_batch()
        if num_samples_so_far % self.batch_size != 0:
            # if there remains an incomplete batch
            batch_num += 1
            process_batch()

    def train(
        self,
        *,
        total_timesteps: Optional[int] = None,
        bco_alpha_T: Optional[int] = 10,
        bco_alpha: float = 0,
        bco_inv_steps: Optional[int] = None,
        bc_n_epochs: Optional[int] = None,
        bc_n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        if total_timesteps is not None:
            assert (
                bco_alpha_T is not None
            ), "Cannot specify both total_timesteps and bco_alpha_T"
            bco_alpha_T = (
                total_timesteps // int(bco_alpha * self.demonstration_size) + 1
            )

        # exploration_steps = self.demonstration_size
        exploration_steps = int(bco_alpha * self.demonstration_size)
        pbar = tqdm(total=total_timesteps, dynamic_ncols=True, unit="step")
        while self._env_steps < total_timesteps:
            print("---")

            print(f"Collecting exploration experience for {exploration_steps} steps")
            sample_until = rollout.make_sample_until(exploration_steps, None)
            trajs = rollout.generate_trajectories(
                self.policy,
                self.venv,
                sample_until,
                rng=self.rng,
                deterministic_policy=False,
            )
            rollout_stats = rollout.rollout_stats(trajs)

            for k, v in rollout_stats.items():
                if "return" in k and "monitor" not in k:
                    self.logger.record(f"rollout/{k}", v)
                if "success" in k:
                    self.logger.record(f"rollout/{k}", v)
            self.logger.dump(self._env_steps)

            transitions = rollout.flatten_trajectories(trajs)
            self.transition_buffer.store(transitions)
            exploration_steps = len(transitions)
            print(f"Collected {exploration_steps} transitions")
            self._env_steps += exploration_steps

            print("Training inverse function")
            infer_ac_losses = self._train_inv_func(
                bco_inv_steps if bco_inv_steps else exploration_steps
            )
            self.logger.record("inv/inv_loss", np.mean(infer_ac_losses))

            print("Training Policy")
            n_batches = bc_n_batches if bc_n_batches else exploration_steps
            self.train_bc(
                n_epochs=bc_n_epochs,
                n_batches=n_batches,
                on_epoch_end=on_epoch_end,
                on_batch_end=on_batch_end,
                log_interval=n_batches,
                log_rollouts_venv=None,
                progress_bar=progress_bar,
                reset_tensorboard=reset_tensorboard,
            )
            pbar.update(exploration_steps)
        pbar.close()
