from dataclasses import dataclass

@dataclass
class Config:
    base_path: str
    img_size: int = 128
    normal_class: str = "notumor"

    # normalization
    norm_mode: str = "global"      # "global" or "local"
    local_block: int = 8           # only for norm_mode="local"

    random_state: int = 42
