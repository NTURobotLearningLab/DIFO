import logging

import sacred

from difo.difo import DiffusionTrainer

diffusion_ingredient = sacred.Ingredient("diffusion")
logger = logging.getLogger(__name__)


@diffusion_ingredient.config
def config():
    demo_batch_size = 64  # Batch size for training on demonstrations
    log_interval = 100
    optimizer_kwargs = None


@diffusion_ingredient.capture
def make_diffusion_trainer(
    *,
    demonstrations,
    reward_net,
    custom_logger,
    demo_batch_size: int,
    optimizer_kwargs,
    log_interval: int,
) -> DiffusionTrainer:
    trainer = DiffusionTrainer(
        demonstrations=demonstrations,
        demo_batch_size=demo_batch_size,
        reward_net=reward_net,
        custom_logger=custom_logger,
        optimizer_kwargs=optimizer_kwargs,
        log_interval=log_interval,
    )
    return trainer
