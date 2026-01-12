"""扩散调度器模块"""

from diffusers import DDPMScheduler, DDIMScheduler
from typing import Optional, Literal


def get_scheduler(
    scheduler_type: str = "ddpm",
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    beta_schedule: str = "scaled_linear",
    prediction_type: str = "epsilon",
) -> DDPMScheduler | DDIMScheduler:
    """
    获取扩散调度器
    
    Args:
        scheduler_type: 调度器类型 ("ddpm" 或 "ddim")
        num_train_timesteps: 训练时间步数
        beta_start: beta 起始值
        beta_end: beta 结束值
        beta_schedule: beta 调度方式
        prediction_type: 预测类型 ("epsilon" 或 "v_prediction")
        
    Returns:
        调度器对象
    """
    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    return scheduler



