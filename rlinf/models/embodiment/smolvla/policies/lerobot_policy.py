from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

class SmolVLAPolicyForRLinf(SmolVLAPolicy):
    def __init__(self, config):
        super().__init__(config)

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs, # Why there is 'temperature'?
    ):
        pass