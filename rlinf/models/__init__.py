# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)

from rlinf.config import SupportedModel, get_supported_model, torch_dtype_from_precision


def get_vla_model_config_and_processor(cfg: DictConfig):
    model_type = get_supported_model(cfg.model.model_type)
    if model_type == SupportedModel.OPENVLA:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        from .embodiment.prismatic.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

        model_config = AutoConfig.from_pretrained(cfg.tokenizer.tokenizer_model)

        dataset_statistics_path = os.path.join(
            cfg.tokenizer.tokenizer_model, "dataset_statistics.json"
        )
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(model_config, "norm_stats", norm_stats)
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )
    elif model_type == SupportedModel.OPENVLA_OFT:
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.prismatic.processing_prismatic import (
            MultiInputPrismaticProcessor as PrismaticProcessorOFT,
        )
        from .embodiment.prismatic.processing_prismatic import PrismaticImageProcessor

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        AutoImageProcessor.register(OpenVLAOFTConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAOFTConfig, PrismaticProcessorOFT)

        model_config = OpenVLAOFTConfig.from_pretrained(
            cfg.tokenizer.tokenizer_model, center_crop=cfg.model.center_crop
        )
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessorOFT.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )

    return model_config, input_processor


def get_model(cfg: DictConfig, override_config_kwargs=None):
    model_path = cfg.model_path
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model_type = get_supported_model(cfg.model_type)
    if model_type == SupportedModel.OPENVLA:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        actor_model_config = OpenVLAConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        from .embodiment.openvla_action_model import OpenVLAForRLActionPrediction

        model = OpenVLAForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            hidden_size=cfg.hidden_size,
            unnorm_key=cfg.unnorm_key,
            config=actor_model_config,
            add_value_head=cfg.add_value_head,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            attn_implementation=cfg.attn_implementation,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            trust_remote_code=cfg.trust_remote_code,
        )

        model.to(torch_dtype)

    elif model_type == SupportedModel.OPENVLA_OFT:
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.openvla_oft_action_model import OpenVLAOFTForRLActionPrediction

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        override_config_kwargs = cfg
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                setattr(actor_model_config, key, val)

        model = OpenVLAOFTForRLActionPrediction.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2",
            config=actor_model_config,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            trust_remote_code=True,
            add_value_head=cfg.add_value_head,
        )

        # oft add
        model.vision_backbone.set_num_images_in_input(cfg.get("num_images_in_input", 1))

        model.to(torch_dtype)

    elif model_type == SupportedModel.OPENPI:
        import glob

        import openpi.shared.download as download
        import openpi.transforms as transforms
        import safetensors
        from openpi.training import checkpoints as _checkpoints

        from .embodiment.openpi import get_openpi_config
        from .embodiment.openpi_action_model import (
            OpenPi0Config,
            OpenPi0ForRLActionPrediction,
        )

        # config
        config_name = getattr(cfg.openpi, "config_name", None)
        actor_train_config = get_openpi_config(config_name)
        actor_model_config = actor_train_config.model
        actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
        override_config_kwargs = cfg.openpi
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                actor_model_config.__dict__[key] = val
        # load model
        checkpoint_dir = download.maybe_download(str(model_path))
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

        model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
            actor_model_config
        )
        # train expert only
        if actor_model_config.train_expert_only:
            model.freeze_vlm()

        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # fsdp replace
        # model.paligemma_with_expert.replace_gemma_decoder_layers()
        # load data stats
        data_config = actor_train_config.data.create(
            actor_train_config.assets_dirs, actor_model_config
        )
        norm_stats = None
        if norm_stats is None:
            # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
            # that the policy is using the same normalization stats as the original training process.
            if data_config.asset_id is None:
                raise ValueError("Asset id is required to load norm stats.")
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir, data_config.asset_id
            )
        # wrappers
        repack_transforms = transforms.Group()
        default_prompt = None
        model.setup_wrappers(
            transforms=[
                *repack_transforms.inputs,
                transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.data_transforms.outputs,
                *repack_transforms.outputs,
            ],
        )

    elif model_type == SupportedModel.MLP_POLICY:
        from .embodiment.mlp_policy import MLPPolicy

        model = MLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
        )
    elif model_type == SupportedModel.GR00T:
        from pathlib import Path

        from rlinf.utils.patcher import Patcher

        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()

        from gr00t.experiment.data_config import load_data_config

        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

        from .embodiment.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction

        if cfg.embodiment_tag == "libero_franka":
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.modality_config:LiberoFrankaDataConfig"
            )
        elif cfg.embodiment_tag == "maniskill_widowx":
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.modality_config:ManiskillWidowXDataConfig"
            )
        else:
            raise ValueError(f"Invalid embodiment tag: {cfg.embodiment_tag}")
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        # The transformer rigisteration is done in gr00t/model/gr00t_n1.py
        model_path = Path(model_path)
        if not model_path.exists():
            # raise error or it triggers auto download from hf(It's cool but we don't have internet connection.)
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,  # This tag determines the state encoder and action head to use
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,  # TODO(lx): unify the embodiment data format and obs converter
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
        model.to(torch_dtype)
        if cfg.rl_head_config.add_value_head:
            # reinitialize the value head after model loading, or there are nan values in the value head after model loading.
            model.action_head.value_head._init_weights()

        if cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)
    elif model_type == SupportedModel.SMOLVLA:
        # TODO: DataConfig; setup wrappers
        from .embodiment.smolvla_action_model import SmolVLAForRLActionPrediction, SmolVLAForRLConfig
        from lerobot.policies.factory import make_pre_post_processors, make_policy
        from lerobot.configs.types import FeatureType, PolicyFeature
        import safetensors
        from collections import OrderedDict
        from rlinf.utils.logging import get_logger
        logger = get_logger()

        
        actor_train_config = SmolVLAForRLConfig()
        actor_model_config = actor_train_config
        actor_model_config.load_vlm_weights = True
        actor_model_config.pretrained_path = 'HuggingFaceVLA/smolvla_libero'
        actor_model_config.vlm_model_name = 'HuggingFaceTB/SmolVLM2-500M-Instruct'
        actor_model_config.expert_width_multiplier = 0.5
        actor_model_config.n_action_steps = cfg.num_action_chunks
        actor_model_config.chunk_size = 1
        # 使用下面两条配置后, 可以完成eval
        actor_model_config.num_vlm_layers = 0
        actor_model_config.prefix_length = 0

        actor_model_config.add_value_head = cfg.add_value_head
        actor_model_config.value_after_vlm = False
        actor_model_config.detach_critic_input = cfg.smolvla.detach_critic_input

        actor_model_config.train_expert_only = True

        # actor_model_config.num_steps = 5
        # actor_model_config.chunk_size = 5
        # actor_model_config.n_action_steps = 5

        # hard coding, to be removed soon
        # actor_model_config.input_features = {
        #     # from 14 to 8
        #     "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        #     "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        # }
        # actor_model_config.output_features = {
        #     "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        # }
        actor_model_config.input_features = {
            'observation.images.image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            'observation.images.image2': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
        actor_model_config.output_features = {
            'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=actor_model_config,
            pretrained_path="HuggingFaceVLA/smolvla_libero",
        )

        # logger.info(f"Using SmolVLA actor_train_config: {actor_train_config}")

        # test load policy
        # pass
        # from lerobot.policies.factory import get_policy_class
        # cls = get_policy_class("smolvla")
        # kwargs = {}
        # kwargs["config"] = actor_model_config
        # kwargs["pretrained_name_or_path"] = "HuggingFaceVLA/smolvla_libero"
        # policy = cls.from_pretrained(**kwargs)
        # for k, v in policy.state_dict().items():
        #     logger.info(f"SmolVLA policy state_dict key: {k}, shape: {v.shape}")

        model = SmolVLAForRLActionPrediction(actor_model_config)
        model.setup_processor(
            preprocessor,
            postprocessor
        )

        if actor_model_config.train_expert_only:
            model.freeze_vlm()

        ckpt_path = "/home/bao_zonghuang/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
        # ckpt_path = "/home/bao_zonghuang/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de"
        # ckpt_path = "/home/bao_zonghuang/codes/python/RLinf/logs/20260430-12:33:13/test_smolvla/checkpoints/global_step_20/actor/model"
        weight_path = os.path.join(ckpt_path, "model.safetensors")
        # weight_path = os.path.join(ckpt_path, "model-00001-of-00001.safetensors")
        
        # logger.info(f"SmolVLA model keys: {model.state_dict().keys()}")
        # logger.info(f"ckpt keys: {safetensors.torch.load_file(weight_path).keys()}")
        ckpt = safetensors.torch.load_file(weight_path)
        model_dict = model.state_dict()
        new_ckpt = OrderedDict()
        mismatched_keys = []
        prefix = "model."
        # prefix = ""
        for k, t in ckpt.items():
            new_k = k[len(prefix):] if k.startswith(prefix) else k
            # new_k = k
            if new_k in model_dict:
                new_ckpt[new_k] = t.clone()  # clone 以防后续 in-place
            else:
                mismatched_keys.append(new_k)
        if len(mismatched_keys) > 0:
            # pass
            logger.warning(f"以下 ckpt keys 未匹配到模型参数: {mismatched_keys}")
        res = model.load_state_dict(new_ckpt, strict=False)
        logger.info(f"SmolVLA load_state_dict result: {res}")
        # logger.info(f"SmolVLA dtype: {list(model.vlm_with_expert.parameters())[0].dtype}, device: {next(model.parameters()).device}")
        model.to(torch_dtype)
    else:
        return None
    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=[
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",
                    "out_proj",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            if model_type == SupportedModel.OPENPI:
                module_to_lora = model.paligemma_with_expert.paligemma
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.paligemma_with_expert.paligemma = module_to_lora
            elif model_type == SupportedModel.SMOLVLA:
                module_to_lora = model.vlm_with_expert.vlm
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.vlm_with_expert.vlm = module_to_lora
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    return model


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
