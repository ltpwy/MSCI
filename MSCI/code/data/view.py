import torch

# 加载模型的 state_dict
state_dict = torch.load("feasibility_mit-states.pt")  # 替换为你的 .pt 文件路径

# 打印 keys 和对应的 shapes
for key, value in state_dict.items():
    print(f"{key}: {value}")
