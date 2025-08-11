import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class CharSequenceDataset(Dataset):
    def __init__(self, input_strs, target_strs, char_to_idx):
        self.input_strs = input_strs
        self.target_strs = target_strs
        self.char_to_idx = char_to_idx
        self.num_chars = len(char_to_idx)
        
    def __len__(self):
        return len(self.input_strs)
    
    def __getitem__(self, idx):
        # 获取输入和目标字符串
        input_str = self.input_strs[idx]
        target_str = self.target_strs[idx]
        
        # 转换为索引序列
        input_data = [self.char_to_idx[c] for c in input_str]
        target_data = [self.char_to_idx[c] for c in target_str]
        
        # 转换为独热编码
        input_one_hot = np.eye(self.num_chars)[input_data]
        
        # 转换为张量
        input_tensor = torch.tensor(input_one_hot, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.long)
        
        return input_tensor, target_tensor

# 数据集：字符序列预测（hello <-> elloh）
# 使用集合确保字符唯一，然后转换为列表
char_set = list(set("hello"))  # 所有可能的唯一字符
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

# 训练数据：双向映射
input_strs = ["hello", "elloh"]
target_strs = ["elloh", "hello"]

# 创建数据集和数据加载器
dataset = CharSequenceDataset(input_strs, target_strs, char_to_idx)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # 批大小为1，打乱数据

# 模型超参数
input_size = len(char_set)
hidden_size = 16
output_size = len(char_set)
num_epochs = 500
learning_rate = 0.01

# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

model = RNNModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练 RNN
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        
        # 初始隐藏状态
        hidden = torch.zeros(1, inputs.size(0), hidden_size)
        
        # 前向传播
        outputs, hidden = model(inputs, hidden)
        
        # 计算损失
        loss = criterion(outputs.reshape(-1, output_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 每50轮打印一次平均损失
    if (epoch + 1) % 50 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# 测试双向预测
def predict_sequence(input_str):
    with torch.no_grad():
        # 准备输入数据
        input_data = [char_to_idx[c] for c in input_str]
        input_one_hot = np.eye(len(char_set))[input_data]
        input_tensor = torch.tensor(input_one_hot, dtype=torch.float32).unsqueeze(0)
        
        # 预测
        test_hidden = torch.zeros(1, 1, hidden_size)
        test_output, _ = model(input_tensor, test_hidden)
        predicted = torch.argmax(test_output, dim=2).squeeze().numpy()
        
        return ''.join([idx_to_char[i] for i in predicted])

# 测试两个方向
test_cases = ["hello", "elloh"]
for test_case in test_cases:
    prediction = predict_sequence(test_case)
    print(f"Input: {test_case}, Predicted: {prediction}")
    
###################################################################################

import mAIn
class test(mAIn.testModel):
    def __init__(self, model: nn.Module, dataloader: mAIn.DataLoader) -> None:
        super().__init__(model, dataloader)
    
    def run(self,inp:str) ->str:
        question= "hello" if n%2==0  else "elloh"
        
