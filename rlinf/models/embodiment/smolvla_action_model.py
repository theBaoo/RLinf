from dataclasses import dataclass, field
from typing import Literal, Any
from typing_extensions import Unpack
import numpy as np
from collections.abc import Sequence
from collections import deque
import torch
import random
import math

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, make_att_2d_masks, resize_with_pad, pad_vector
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger


def _log_shapes(obj, logger, name="root"):
    """Recursively log keys and shapes/types for dict/list/torch/ndarray objects.

    Keeps output small: prints type and shape/length only.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(obj, dict):
        for k, v in obj.items():
            _log_shapes(v, logger, f"{name}.{k}")
    elif isinstance(obj, (list, tuple)):
        logger.info(f"{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:10]):
            _log_shapes(v, logger, f"{name}[{i}]")
        if len(obj) > 10:
            logger.info(f"{name}: ... (showing first 10 of {len(obj)})")
    elif 'torch' in globals() and torch.is_tensor(obj):
        try:
            logger.info(f"{name}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}")
        except Exception:
            logger.info(f"{name}: Tensor (unable to read shape)")
    elif _np is not None and isinstance(obj, _np.ndarray):
        logger.info(f"{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        # fallback for scalars / other objects
        tname = type(obj).__name__
        # avoid printing large reprs
        short = repr(obj)
        if len(short) > 200:
            short = short[:200] + "..."
        logger.info(f"{name}: {tname} repr={short}")

@dataclass
class SmolVLAForRLConfig(SmolVLAConfig):
    """
    Configuration class for SmolVLA model adapted for reinforcement learning tasks.
    Copied from OpenPi's config.
    """
    noise_level: float = 0.2
    action_chunk: int = 5
    train_expert_only: bool = False
    action_env_dim: int = 7  # for libero
    num_steps: int = 10
    noise_method: str = "flow_cps" # "reinflow"  # flow_sde, flow_cps
    safe_get_logprob: bool = False
    joint_logprob: bool = False
    double_layer: bool = False
    detach_critic_input: bool = False
    chunk_critic_input: bool = False
    add_value_head: bool = False
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    ignore_last: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token
    simulator_type: str = "libero"  # libero, maniskill, robotwin, metaworld
    cast_inputs_to_model_dtype: bool = True
    debug_logprob_norm: bool = False
    debug_logprob_norm_interval: int = 200

class SmolVLAForRLActionPrediction(VLAFlowMatching):
    """
    SmolVLA for reinforcement learning action prediction.
    """

    def __init__(
        self,
        config: SmolVLAForRLConfig,
    ):
        super().__init__(config)
        self.config = config

        # rl model init
        if self.config.value_after_vlm:
            assert not self.config.joint_logprob, (
                "joint logprob is not tested for value_after_vlm mode"
            )
            assert not (self.config.double_layer and self.config.joint_logprob), (
                "double_layer and joint_logprob can not be set at the same time"
            )
            proj_width = 480 # smolvla, 720 to 480
        else:
            proj_width = 480 # smolvla

        self.global_step = 0
        self._logprob_norm_debug_calls = 0

        if self.config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            # self.value_head.to(torch.bfloat16)

        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )

        if self.config.noise_method == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=proj_width,
                out_dim=self.config.max_action_dim, # action_dim in openpi config
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

        # store action chunks
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def set_global_step(self, step):
        self.global_step = step

    def _get_model_dtype(self) -> torch.dtype:
        if hasattr(self, "action_in_proj") and hasattr(self.action_in_proj, "weight"):
            return self.action_in_proj.weight.dtype
        return next(self.parameters()).dtype

    def _cast_float_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            return x
        if not x.is_floating_point():
            return x
        if not getattr(self.config, "cast_inputs_to_model_dtype", True):
            return x
        target_dtype = self._get_model_dtype()
        if x.dtype == target_dtype:
            return x
        return x.to(dtype=target_dtype)

    def _cast_float_tensor_list(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._cast_float_tensor(x) for x in xs]

    def setup_processor(
        self,
        preprocessor,
        postprocessor,
    ):
        # TODO: 迁移SmolVLA的processor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    # TODO: 迁移SmolVLA的processor
    def input_processor(
        self,
        env_obs
    ):
        """
        RLinf's env_obs: {
            "images": image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": task_descriptions,
        }

        SmolVLA's batch: {
            "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
            "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
            "task": ["Pick up the object"],
        }
        """
        # logger = get_logger()
        # logger.info("--- debug: env_obs shapes ---")
        # _log_shapes(env_obs, logger, "env_obs")
        
        to_processed_obs = {
            "observation.state": env_obs["states"],
            # "observation.images.base_0_rgb": env_obs["images"],
            "observation.images.image": env_obs["images"],
            "observation.images.image2": env_obs["wrist_images"],
            "task": env_obs["task_descriptions"]
        }
        # logger.info("--- debug: to_processed_obs shapes ---")
        # _log_shapes(to_processed_obs, logger, "to_processed_obs")

        # RLinf does the same
        # LiberoProcessorStep in lerobot:
        # image = to_processed_obs["observation.images.image"]
        # image = torch.flip(image, dims=[2, 3])
        # to_processed_obs["observation.images.image"] = image

        # lerobot: preprocess_observation
        img = to_processed_obs["observation.images.image"]
        img = img.type(torch.float32)
        img /= 255
        to_processed_obs["observation.images.image"] = img

        wimg = to_processed_obs["observation.images.image2"]
        wimg = wimg.type(torch.float32)
        wimg /= 255
        to_processed_obs["observation.images.image2"] = wimg

        processed_obs = self.preprocessor(to_processed_obs)

        # logger.info("--- debug: processed_obs shapes ---")
        # _log_shapes(processed_obs, logger, "processed_obs")
        # ignore self.config.adapt_to_pi_aloha
        return processed_obs

    def input_transform(
        self
    ):
        pass

    # TODO: 迁移SmolVLA的processor
    def output_transform(
        self
    ):
        pass
    
    """
    Copied from lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py: SmolVLAPolicy
    """
    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            # ensure image tensor is float in [0, 1] before interpolation/resizing
            # img = img.type(torch.float32)
            # img /= 255
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    """
    Copied from lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py: SmolVLAPolicy
    """
    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        state = self._cast_float_tensor(state)
        return state

    def forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs
    ):
        compute_values = kwargs.get("compute_values", False)
        chains = self._cast_float_tensor(data["chains"])
        denoise_inds = data["denoise_inds"]

        images, img_masks, lang_tokens, lang_masks, state = self.preprocess_for_train(data)
        images = self._cast_float_tensor_list(images)
        state = self._cast_float_tensor(state)

        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)

        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.n_action_steps, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.n_action_steps, : self.config.action_env_dim
        ]

        # post process
        # if self.config.joint_logprob:
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            # "prev_logprobs": prev_logprobs,
            "values": value_t,
            "entropy": entropy,
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs, # Why there is 'temperature'?
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # if len(self._queues[ACTION]) == 0:
        processed_obs = self.input_processor(env_obs)
        images, img_masks = self.prepare_images(processed_obs)
        lang_tokens = processed_obs[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = processed_obs[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(processed_obs)
        images = self._cast_float_tensor_list(images)
        state = self._cast_float_tensor(state)
        # call sample_actions with keyword args so positional 'mode' is not interpreted as 'noise'
        outputs = self.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            mode=mode,
            compute_values=compute_values,
            **kwargs,
        )

        # unpad actions
        actions = outputs["actions"][:, : self.config.n_action_steps, :]
        # actions = outputs[:, :self.config.n_action_steps, :]

        # logger.info("--- debug: raw chunk actions shapes ---")
        # _log_shapes(actions, logger, "raw chunk actions")

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        # logger.info("--- debug: unpad chunk actions shapes ---")
        # _log_shapes(actions, logger, "unpad chunk actions")

        # actions = actions.transpose(0, 1)[: self.config.n_action_steps]
        # self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        # actions = self._queues[ACTION].popleft()

        # actions: [bsize, chunk_size, action_dim]
        # actions = self.postprocessor(actions)
        post_actions = []
        for i in range(actions.shape[1]):
            chunk = actions[:, i, :]
            pa = self.postprocessor(chunk)
            post_actions.append(pa)
        actions = torch.stack(post_actions, dim=1)
        # actions = actions.transpose(0, 1)[: self.config.n_action_steps]
        # logger.info("--- debug: postprocessed actions shapes ---")
        # _log_shapes(actions, logger, "postprocessed actions")
        # logger.info(f"actions: {actions}")

        # 字段命名?
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            # OpenPI-like: keep raw obs tensors for memory efficiency.
            "obs_images": env_obs["images"],
            "obs_wrist_images": env_obs["wrist_images"],
            "obs_states": env_obs["states"],
            # Keep rollout language tokens for deterministic train-time logprob/value.
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            # Still pass raw task text for preprocessing parity in forward.
            "task_descriptions": env_obs["task_descriptions"],
        }
        if self.config.simulator_type == "libero":
            # forward_inputs["observations/wrist_image"] = env_obs["wrist_images"]
            pass

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        # from rlinf.utils.logging import get_logger
        # logger = get_logger()
        # logger.info(f"chunk start")
        # for _ in range(actions.shape[1]):
        #     logger.info(f"gripper: {actions[0, _, -1].item()}")
        # logger.info(f"chunk end")

        return actions, result

    """
    OpenPI: 传入obs, 再调用内部处理函数获取img等信息
    SmolVLA: 直接传入img等信息
    """
    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        mode="train",
        compute_values=True,
        **kwargs, # temperature???
    ):
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device
        num_steps = self.config.num_steps

        if noise is None:
            action_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(action_shape, device)
        noise = self._cast_float_tensor(noise)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        pre_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # SmolVLA似乎不需要这一步
        # pre-att-2d-masks to 4d
        # pre_att_2d_masks = self.prepare_attention_masks_4d(pre_att_2d_masks)

        (prefix_output, _), past_key_values = self.vlm_with_expert.forward(
            attention_mask=pre_att_2d_masks,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None],
            past_key_values=None,
            use_cache=True,
            fill_kv_cache=True,
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        # SmolVLA???
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "eval":
            denoise_inds = torch.tensor([-1] * num_steps)
        elif mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                # if self.config.noise_method == "reinflow":
                #     denoise_inds = torch.tensor([random.randint(0, num_steps - 1)] * num_steps)
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise_inds
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(
                x_t,
                x_t_mean,
                x_t_std,
            )
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.n_action_steps, : self.config.action_env_dim
        ]

        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]

        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    
    def sample_mean_var_val(
        self,
        x_t, # 当前噪声/状态, shape与action_out_proj一致
        idx, # int -> [idx] * bsize; tensor
        # conditioning inputs
        state,
        prefix_pad_masks,
        past_key_values,

        mode, # "train" or "eval"
        denoise_steps,
        compute_values=True,
        prefix_embs=None,
        prefix_att_masks=None,
        use_train_forward=False,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expend the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        x_t = self._cast_float_tensor(x_t)
        # build parameters
        if self.config.noise_anneal:
            # 线性插值
            noise_start, noise_end, noise_anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, noise_anneal_steps)
                / noise_anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(
            1,
            1 / denoise_steps,
            denoise_steps,
            device=device
        )
        timesteps = torch.concat(
            [timesteps, torch.tensor([0.0], device=device)]
        ) # shape: [denoise_steps + 1], from 1 to 0
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1] # 离散时间步步长
        # velocity prediction
        if use_train_forward:
            if prefix_embs is None or prefix_att_masks is None:
                raise ValueError(
                    "prefix_embs and prefix_att_masks are required when use_train_forward=True"
                )
            suffix_out = self.get_suffix_out_train(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_t,
                t_input,
            )
        else:
            suffix_out = self.get_suffix_out(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_input,
            )
        v_t = self.action_out_proj(suffix_out)  # 预测的速度场, [bs, n_action_steps, max_action_dim]
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)  # [bs, dim]
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value).squeeze(-1)  # [bs]
        else:
            value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input) # ?
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "reinflow":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.reinflow_explore_noise_net(
                    suffix_out,
                )
            elif self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            else:
                raise ValueError(
                    f"Invalid noise method: {self.config.noise_method}"
                )
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def prepare_attention_masks_4d(self, att_2d_masks):
        """
        Helper method to prepare 4D attention masks for transformer.
        Copied from OpenPI's implementation.
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks, # [bsize, prefix_len], bool
        past_key_values,
        x_t,
        timestep,
    ):
        """
        Apply one denoising step of the noise `x_t` at a given timestep.
        把当前噪声动作 x_t（作为 decoder / suffix 的输入 token），在给定 prefix 的条件下，经模型解码一次，得到 decoder 在 suffix 位置上的隐藏表示（suffix_out）。这些隐藏表示随后被 action_out_proj 投影成速度场 v_t（或直接投影为动作相关参数）。
        """

        # openpi's 额外传入state, 额外返回adarms_cond
        # Normalize `timestep` to a 1D float tensor of shape (batch_size,) as
        # expected by create_sinusoidal_pos_embedding. Accepts:
        # - Python scalar (int/float)
        # - 0-dim tensor
        # - 1-dim tensor of length batch_size
        # - higher-dim tensor that can be squeezed/reshaped to (batch_size,)
        # batch_size = prefix_pad_masks.shape[0]
        # device = state.device if torch.is_tensor(state) else None

        # if not torch.is_tensor(timestep):
        #     # scalar or python object -> fill batch
        #     try:
        #         tval = float(timestep)
        #     except Exception:
        #         tval = 0.0
        #     timestep = torch.full((batch_size,), tval, dtype=torch.float32, device=device)
        # else:
        #     # coerce dtype/device and shape
        #     timestep = timestep.to(device=device, dtype=torch.float32)
        #     if timestep.ndim == 0:
        #         timestep = timestep.expand(batch_size)
        #     elif timestep.ndim == 1 and timestep.shape[0] == batch_size:
        #         pass
        #     else:
        #         try:
        #             timestep = timestep.reshape(batch_size)
        #         except Exception:
        #             timestep = timestep.squeeze()
        #             if timestep.ndim == 0:
        #                 timestep = timestep.expand(batch_size)

        suffix_embs, suffix_pad_masks, suffix_att_masks = (
            self.embed_suffix(
                x_t,
                timestep,
            )
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        # prefix_len should come from prefix_pad_masks passed into this function
        prefix_len = prefix_pad_masks.shape[1]

        # build 2d mask of shape [B, suffix_len, prefix_len] from prefix_pad_masks
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        # make_att_2d_masks expects (pad_masks[B,N], att_masks[B,N]).
        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
        )

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks],
            dim=2,
        )

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        postion_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # SmolVLA似乎不需要这一步
        # Prepare attention masks
        # full_att_2d_masks_4d = self.
        # full_att_2d_masks = self.prepare_attention_masks_4d(full_att_2d_masks)

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=postion_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            # With non-empty past_key_values, use_cache must stay enabled so
            # key/value states are concatenated with cached prefix states.
            use_cache=True,
            fill_kv_cache=False,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = self._cast_float_tensor(suffix_out)

        return suffix_out

    def get_suffix_out_train(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_t,
        timestep,
    ):
        """Training-only suffix forward.

        Runs a single full forward with [prefix, suffix] embeddings to avoid
        suffix-only cache constraints in SmolVLMWithExpert.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, timestep
        )
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = self._cast_float_tensor(suffix_out)
        return suffix_out

    def get_logprob_norm(self, sample, mu, sigma):
        """
        Copied from OpenPi's implementation.

        logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        """
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
            constant_term = None
            exponent_term = None
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)

        if getattr(self.config, "debug_logprob_norm", False):
            self._logprob_norm_debug_calls += 1
            interval = max(1, int(getattr(self.config, "debug_logprob_norm_interval", 200)))
            if self._logprob_norm_debug_calls % interval == 0:
                logger = get_logger()

                def _brief(name, x):
                    x = x.detach().float()
                    return (
                        f"{name}(shape={tuple(x.shape)},mean={x.mean().item():.6g},"
                        f"std={x.std(unbiased=False).item():.6g},min={x.min().item():.6g},max={x.max().item():.6g})"
                    )

                parts = [
                    _brief("sample", sample),
                    _brief("mu", mu),
                    _brief("sigma", sigma),
                    _brief("log_prob", log_prob),
                ]
                if constant_term is not None and exponent_term is not None:
                    parts.extend(
                        [
                            _brief("constant_term", constant_term),
                            _brief("exponent_term", exponent_term),
                        ]
                    )
                logger.info("[DBG-LOGPROB-NORM] " + " | ".join(parts))
        return log_prob
    
    def preprocess_for_train(self, data):
        if (
            "obs_images" in data
            and "obs_wrist_images" in data
            and "obs_states" in data
        ):
            to_processed_obs = {
                "observation.state": data["obs_states"],
                "observation.images.image": data["obs_images"].to(torch.float32)
                / 255.0,
                "observation.images.image2": data["obs_wrist_images"].to(torch.float32)
                / 255.0,
                "task": data.get("task_descriptions", ""),
            }
        else:
            raise KeyError(
                "SmolVLA training forward expects raw obs tensors: obs_images/obs_wrist_images/obs_states"
            )

        # Keep training-time preprocessing path aligned with rollout by always invoking preprocessor.
        # We intentionally discard regenerated language tokens to avoid rollout/train token drift.
        processed_obs = self.preprocessor(to_processed_obs)

        if OBS_LANGUAGE_TOKENS in data and OBS_LANGUAGE_ATTENTION_MASK in data:
            lang_tokens = data[OBS_LANGUAGE_TOKENS]
            lang_masks = data[OBS_LANGUAGE_ATTENTION_MASK]
        elif "lang_tokens" in data and "lang_masks" in data:
            lang_tokens = data["lang_tokens"]
            lang_masks = data["lang_masks"]
        elif OBS_LANGUAGE_TOKENS in processed_obs and OBS_LANGUAGE_ATTENTION_MASK in processed_obs:
            # Fallback only when rollout did not carry language tokens.
            lang_tokens = processed_obs[OBS_LANGUAGE_TOKENS]
            lang_masks = processed_obs[OBS_LANGUAGE_ATTENTION_MASK]
        else:
            raise KeyError(
                "SmolVLA training forward expects rollout lang_tokens/lang_masks"
            )

        images, img_masks = self.prepare_images(processed_obs)
        state = self.prepare_state(processed_obs)
        images = self._cast_float_tensor_list(images)
        state = self._cast_float_tensor(state)

        return images, img_masks, lang_tokens, lang_masks, state
    
    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        chains = self._cast_float_tensor(chains)
        state = self._cast_float_tensor(state)
        images = self._cast_float_tensor_list(images)
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        if self.use_vlm_value:
            pre_att_2d_masks = make_att_2d_masks(
                prefix_pad_masks, prefix_att_masks
            )
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            [prefix_output, _], _ = self.vlm_with_expert.forward(
                attention_mask=pre_att_2d_masks,
                position_ids=prefix_position_ids,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
            )
        else:
            prefix_output = None
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_nxt = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values=None,
                mode="train",
                denoise_steps=self.config.num_steps,
                compute_values=compute_values,
                prefix_embs=prefix_embs,
                prefix_att_masks=prefix_att_masks,
                use_train_forward=True,
            )
            log_probs = self.get_logprob_norm(
                chains_nxt,
                x_t_mean,
                x_t_std,
            )
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        if self.config.noise_method == "reinflow":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output.shape: [1, 141, 960]
        # TODO: 获取SmolVLA的下两个数值
        lang_length = 0
        all_length = 0
        if self.config.simulator_type == "libero":
            camera_num = 2
        else:
            raise ValueError(
                f"Invalid simulator type: {self.config.simulator_type}; only libero is supported for SmolVLA."
            )
        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * camera_num
                + [False] * 256 * (3 - camera_num)
                + [False] * lang_length
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = self._cast_float_tensor(prefix_out_value)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.vlm_with_expert.vlm.eval()
            for param in self.vlm_with_expert.vlm.parameters():
                param.requires_grad = False