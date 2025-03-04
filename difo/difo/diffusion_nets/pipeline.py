from typing import List, Optional, Tuple, Union

import einops
import torch
from diffusers import DDPMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class DDPMConditionPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"
    scheduler: DDPMScheduler

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        condition: int,
        condition_embs: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if getattr(self.unet.config, "sample_size", None) is None:
            # image_shape = (batch_size, self.unet.config.input_size)
            image_shape = (batch_size, 8)
        elif isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

            if condition_embs is not None:
                if not isinstance(condition_embs, torch.Tensor):
                    condition_embs = torch.tensor(condition_embs)
                condition_embs = einops.repeat(
                    condition_embs, "1 ... -> b ...", b=batch_size
                )
                condition_embs = condition_embs.to(self.device)
            condition = torch.full(
                (batch_size,), condition, device=self.device, dtype=torch.long
            )

        if image.dtype != self.unet.dtype:
            image = image.to(self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            noisy_image = image

            # 1. predict noise model_output
            if condition_embs is not None:
                model_output = self.unet(
                    sample=noisy_image,
                    timestep=t,
                    class_labels=condition,
                    condition_emb=condition_embs,
                ).sample
            else:
                model_output = self.unet(noisy_image, t, condition).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        if len(image.shape) == 4:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = image.cpu()
            if output_type == "np":
                image = image.numpy()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
