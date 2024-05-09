from typing import Optional, List

from llm2vec.models import MistralBiForMNTP, LlamaBiForMNTP
from peft import LoraConfig, get_peft_model


def get_model_class(config):
    config_class_name = config.__class__.__name__
    if config_class_name == "MistralConfig":
        return MistralBiForMNTP
    elif config_class_name == "LlamaConfig":
        return LlamaBiForMNTP
    else:
        raise ValueError(f"Model class {config_class_name} not supported.")

def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )
    # model organization is MODEL_TYPEBiForMNTP.model -> MODEL_TYPELBiModel, we have to apply PEFT to the inner model
    peft_model = get_peft_model(model.get_model_for_peft(), config)
    print(f"Model's Lora trainable parameters:")
    peft_model.print_trainable_parameters()
    model.set_model_for_peft(peft_model)
    return model