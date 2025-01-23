import torch

# 加载模型
model = torch.load('./data/feasibility_mit-states.pt')
feasibility = model['feasibility']

# 打印张量的形状
print(feasibility.shape)
# 打印模型结构

