# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import gradio as gr
import regex as re
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from transformers import AutoTokenizer, UMT5EncoderModel


def get_sigmas(scheduler, timesteps, dtype=torch.float32, device="cuda"):
    # Ensure device is available before using it
    if isinstance(device, str) and device.startswith("cuda"):
        if not torch.cuda.is_available():
            device = "cpu"
    
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    return sigma

class WanT2TexPipeline(WanPipeline):
    def __init__(self, tokenizer, text_encoder, transformer, vae, scheduler):
        super().__init__(tokenizer, text_encoder, transformer, vae, scheduler)
        self.uv_scheduler = copy.deepcopy(scheduler)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        treat_as_first: Optional[bool] = True,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        
        ####################
        if treat_as_first:
            num_latent_frames = num_frames // self.vae_scale_factor_temporal
        else:
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        ####################

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        device: Optional[str] = "cuda",

        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        cond_model_latents: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        uv_height=None,
        uv_width=None,
        uv_num_frames=None,
        # multi_task_cond=None,
        treat_as_first=True,
        gt_condition:Tuple[Optional[Float[Tensor, "B C F H W"]], Optional[Float[Tensor, "B C F H W"]]]=None,
        inference_img_cond_frame=None,
        use_qk_geometry=False,
        task_type="all",
        progress=None
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # ATTENTION: My inputs are images, so the num_frames is 5, without time dimension compression.
        # if num_frames % self.vae_scale_factor_temporal != 1:
        #     raise ValueError(
        #         f"num_frames should be divisible by {self.vae_scale_factor_temporal} + 1, but got {num_frames}."
        #     )
        #     num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        # num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # --- device / offload handling ---------------------------------------
        device = torch.device(device) if isinstance(device, str) else device

        # Do NOT move the whole pipeline here. Offload (if enabled) will handle submodules.
        # Instead, record an execution device and use it for downstream tensor allocations.
        exec_device = getattr(self, "_execution_device", None)
        if exec_device is None:
            exec_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self._execution_device = exec_device
            except Exception:
                pass
        device = exec_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.uv_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        mv_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            treat_as_first=treat_as_first,
        )
        uv_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            uv_height,
            uv_width,
            uv_num_frames,
            torch.float32,
            device,
            generator,
            treat_as_first=True # UV latents are always different from the others, so treat as the first frame
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        # with progress.tqdm(total=num_inference_steps, desc="Diffusing...") as progress_bar:
        for i, t in enumerate(tqdm(timesteps, desc="Diffusing...")):
            if self.interrupt:
                continue

            # set conditions
            timestep_df = torch.ones((batch_size, num_frames // self.vae_scale_factor_temporal + 1)).to(device) * t
            sigmas = get_sigmas(self.scheduler, rearrange(timestep_df, "B F -> (B F)"), dtype=transformer_dtype, device=device)
            sigmas = rearrange(sigmas, "(B F) -> B 1 F 1 1", B=batch_size)
            match task_type:
                case "geo+mv2tex":
                    timestep_df[:, :num_frames // self.vae_scale_factor_temporal] = self.min_noise_level_timestep
                    sigmas[:, :, :num_frames // self.vae_scale_factor_temporal, ...] = self.min_noise_level_sigma
                    mv_noise = torch.randn_like(mv_latents) # B C 4 H W
                    mv_latents = (1.0 - sigmas[:, :, :-1, ...]) * gt_condition[0] + sigmas[:, :, :-1, ...] * mv_noise 
                case "img2tex":
                    assert inference_img_cond_frame is not None, "inference_img_cond_frame should be specified for img2tex task"
                    # Use specified frame index as condition instead of just first frame
                    timestep_df[:, inference_img_cond_frame: inference_img_cond_frame + 1] = self.min_noise_level_timestep
                    sigmas[:, :, inference_img_cond_frame: inference_img_cond_frame + 1, ...] = self.min_noise_level_sigma
                    mv_noise = randn_tensor(mv_latents[:, :, inference_img_cond_frame: inference_img_cond_frame + 1].shape, generator=generator, device=device, dtype=self.dtype)
                    # mv_noise = torch.randn_like(mv_latents[:, :, inference_img_cond_frame: inference_img_cond_frame + 1], generator=generator) # B C selected_frames H W
                    mv_latents[:, :, inference_img_cond_frame: inference_img_cond_frame + 1, ...] = (1.0 - sigmas[:, :, inference_img_cond_frame: inference_img_cond_frame + 1, ...]) * gt_condition[0] + sigmas[:, :, inference_img_cond_frame: inference_img_cond_frame + 1, ...] * mv_noise 
                case "soft_render":
                    timestep_df[:, -1:] = self.min_noise_level_timestep
                    sigmas[:, :, -1:, ...] = self.min_noise_level_sigma
                    uv_noise = torch.randn_like(uv_latents) # B C 1 H W
                    uv_latents = (1.0 - sigmas[:, :, -1:, ...]) * gt_condition[1] + sigmas[:, :, -1:, ...] * uv_noise 
                case "geo2mv":
                    timestep_df[:, -1:] = 1000.
                    sigmas[:, :, -1:, ...] = 1.
                case _:
                    pass
            
            # add geometry information to channel C
            mv_latents_input = torch.cat([mv_latents, cond_model_latents[0]], dim=1)
            uv_latents_input = torch.cat([uv_latents, cond_model_latents[1]], dim=1)
            if self.do_classifier_free_guidance:
                mv_latents_input = torch.cat([mv_latents_input, mv_latents_input], dim=0)
                uv_latents_input = torch.cat([uv_latents_input, uv_latents_input], dim=0)

            self._current_timestep = t
            latent_model_input = (mv_latents_input.to(transformer_dtype), uv_latents_input.to(transformer_dtype))
            # timestep = t.expand(mv_latents.shape[0])

            noise_out = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_df,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                # task_cond=multi_task_cond,
                return_dict=False,
                use_qk_geometry=use_qk_geometry
            )[0]
            mv_noise_out, uv_noise_out = noise_out

            if self.do_classifier_free_guidance:
                mv_noise_uncond, mv_noise_pred = mv_noise_out.chunk(2)
                uv_noise_uncond, uv_noise_pred = uv_noise_out.chunk(2)
                mv_noise_pred = mv_noise_uncond + guidance_scale * (mv_noise_pred - mv_noise_uncond)
                uv_noise_pred = uv_noise_uncond + guidance_scale * (uv_noise_pred - uv_noise_uncond)
            else:
                mv_noise_pred = mv_noise_out
                uv_noise_pred = uv_noise_out

            # compute the previous noisy sample x_t -> x_t-1
            # The conditions will be replaced anyway, so perhaps we don't need to step frames seperately
            mv_latents = self.scheduler.step(mv_noise_pred, t, mv_latents, return_dict=False)[0]
            uv_latents = self.uv_scheduler.step(uv_noise_pred, t, uv_latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                raise NotImplementedError()
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # # call the callback, if provided
            # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            # video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = (mv_latents, uv_latents)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
