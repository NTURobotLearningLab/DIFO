import torch as th
from torch.nn import functional as F

from imitation.rewards import reward_nets


class GAILRewardNetFromDiscriminatorLogit(reward_nets.RewardNet):
    r"""Converts the discriminator logits raw value to a reward signal.

    Wrapper for reward network that takes in the logits of the discriminator
    probability distribution and outputs the corresponding reward for the GAIL
    algorithm.

    Below is the derivation of the transformation that needs to be applied.

    The GAIL paper defines the cost function of the generator as:

    .. math::

        \log{D}

    as shown on line 5 of Algorithm 1. In the paper, :math:`D` is the probability
    distribution learned by the discriminator, where :math:`D(X)=1` if the trajectory
    comes from the generator, and :math:`D(X)=0` if it comes from the expert.
    In this implementation, we have decided to use the opposite convention that
    :math:`D(X)=0` if the trajectory comes from the generator,
    and :math:`D(X)=1` if it comes from the expert. Therefore, the resulting cost
    function is:

    .. math::

        \log{(1-D)}

    Since our algorithm trains using a reward function instead of a loss function, we
    need to invert the sign to get:

    .. math::

        R=-\log{(1-D)}=\log{\frac{1}{1-D}}

    Now, let :math:`L` be the output of our reward net, which gives us the logits of D
    (:math:`L=\operatorname{logit}{D}`). We can write:

    .. math::

        D=\operatorname{sigmoid}{L}=\frac{1}{1+e^{-L}}

    Since :math:`1-\operatorname{sigmoid}{(L)}` is the same as
    :math:`\operatorname{sigmoid}{(-L)}`, we can write:

    .. math::

        R=-\log{\operatorname{sigmoid}{(-L)}}

    which is a non-decreasing map from the logits of D to the reward.
    """

    def __init__(self, base: reward_nets.RewardNet):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.base.forward(state, action, next_state, done)
        return -F.logsigmoid(-logits)


class AIRLRewardNetFromDiscriminatorLogit(reward_nets.RewardNet):
    r"""Converts the discriminator logits raw value to a reward signal.

    Wrapper for reward network that takes in the logits of the discriminator
    probability distribution and outputs the corresponding reward for the AIRL
    algorithm.
    """

    def __init__(self, base: reward_nets.RewardNet):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.base.forward(state, action, next_state, done)
        return F.logsigmoid(logits) - F.logsigmoid(-logits)


class AIRLPositiveRewardNetFromDiscriminatorLogit(reward_nets.RewardNet):
    r"""Converts the discriminator logits raw value to a reward signal.

    Wrapper for reward network that takes in the logits of the discriminator
    probability distribution and outputs the corresponding reward for the AIRL
    algorithm.
    """

    def __init__(self, base: reward_nets.RewardNet):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.base.forward(state, action, next_state, done)
        return F.logsigmoid(logits) - F.logsigmoid(-logits) + 20


class DenoisingRewardNetFromMSE(reward_nets.RewardNet):
    def __init__(self, base: reward_nets.RewardNet):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        mse_loss = self.base.forward(state, action, next_state, done)
        return -mse_loss


REWARD_FN_DICT = {
    "GAIL": GAILRewardNetFromDiscriminatorLogit,
    "AIRL": AIRLRewardNetFromDiscriminatorLogit,
    "AIRL_positive": AIRLPositiveRewardNetFromDiscriminatorLogit,
}
