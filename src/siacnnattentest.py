import torch
import torch.nn as nn
import pandas as pd
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, input1, input2):
        energy = torch.tanh(self.attention(torch.cat((input1, input2), dim=1)))
        attention_weights = torch.softmax(energy, dim=1)
        return attention_weights
# 定义Siamese神经网络模型
class SiameseCNNAttention(nn.Module):
    def __init__(self):
        super(SiameseCNNAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.attention = Attention(64)

    def forward_one(self, x):
        x = torch.relu(self.conv1(x.unsqueeze(0).unsqueeze(0)))  # 添加batch和channel维度
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        attention_weights = self.attention(output1, output2)
        output = torch.abs(output1 - output2) * attention_weights
        output = self.sigmoid(self.fc3(output))
        return output.squeeze().item()  # 返回预测的相似性概率值

# 加载保存的最佳模型参数
model = SiameseCNNAttention()
model.load_state_dict(torch.load("best_model11_precision_siamese_cnn_attention.pth"))
model.eval()

# 输入两个编码
input1 = "4104B30304G"  # 示例编码1
input2 = "4104244241G"  # 示例编码2

# 数据预处理
def process_code(code):
    code = [int(c) if c.isdigit() else ord(c.upper()) - ord('A') + 10 for c in code]
    return torch.tensor(code)

input1_tensor = process_code(input1)
input2_tensor = process_code(input2)

# 进行预测
output = model(input1_tensor.float(), input2_tensor.float())
print(output)

if output > 0.5:
    print("相似")
else:
    print("不相似")
