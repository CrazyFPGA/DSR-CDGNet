import yaml
from pathlib import Path

def load_config(config_path='config.yaml'):
    """加载并返回YAML配置文件内容。"""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(path, 'r', encoding='utf-8') as f: # 明确使用utf-8编码
        config = yaml.safe_load(f)
    
    # 根据 TARGET_FAULT_CLASS 自动设置相关参数
    target_class = config.get('TARGET_FAULT_CLASS')
    if target_class == 8:
        # 任务8：1024@0.002
        config['hyperparameters']['HIGH_PERFORMANCE_BATCH_SIZE'] = 1024
        # 如果是不采用PAA的话，数据量大，采用3072
        #config['hyperparameters']['HIGH_PERFORMANCE_BATCH_SIZE'] = 3072
        config['hyperparameters']['learning_rate'] = 0.002
        #config['hyperparameters']['learning_rate'] = 0.008
    elif target_class == 2:
        # 任务2：256@0.001
        config['hyperparameters']['HIGH_PERFORMANCE_BATCH_SIZE'] = 256
        config['hyperparameters']['learning_rate'] = 0.001
    
    return config

# 加载一次，作为全局配置使用
config = load_config()
