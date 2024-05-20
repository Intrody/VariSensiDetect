import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
#Best Precision: 0.9207129686539643
#Best F1 Score: 0.9245056920311564
#Best Precision: 0.7741424802110818
#Best F1 Score: 0.8342919748014243
#=======================================
# 定义数据集类W
# lr=0.0001 Epoch 483, Loss: 0.0971789289103291, Accuracy: 0.9337716094472851, Precision: 0.8127490039840638, Recall: 0.8691418137553256, F1 Score: 0.8400000000000001
# lr=0.001 无删除 Epoch 490, Loss: 0.058533134152796644, Accuracy: 0.9301193084976869, Precision: 0.7681886603110888, Recall: 0.9318320146074255, F1 Score: 0.8421342134213422
# lr-0.001 删除 Epoch 373, Loss: 0.06748959741919643, Accuracy: 0.9358412466520575, Precision: 0.7983957219251336, Recall: 0.9087035909920876, F1 Score: 0.8499857671505834
# 平衡数据集 Epoch 689, Loss: 0.054522192507083046, Accuracy: 0.9177842565597668, Precision: 0.9092002405291641, Recall: 0.9202678027997565, F1 Score: 0.9147005444646098
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, usecols=[1, 3, 4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input1 = self.process_code(str(self.data.iloc[idx, 0]))
        input2 = self.process_code(str(self.data.iloc[idx, 1]))
        label = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float)
        return input1, input2, label

    def process_code(self, code):
        code = [int(c) if c.isdigit() else ord(c.upper()) - ord('A') + 10 for c in code]
        code = [c if c != 0 else -1 for c in code]  # 将编码中的0替换为-1，以避免在判断相似性时被考虑

        return torch.tensor(code)

# 定义注意力机制
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
        x = torch.relu(self.conv1(x.unsqueeze(1)))
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
        return output

if __name__ == '__main__':

    # 准备数据集和数据加载器
    train_dataset = CustomDataset("updated_dataset_train.csv")
    test_dataset = CustomDataset("updated_dataset_test.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = SiameseCNNAttention()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    best_precision = 0.0
    best_f1 = 0.0
    for epoch in range(700):  # 你可能需要更多的epoch数
        running_loss = 0.0
        model.train()  # 将模型设为训练模式
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1.float(), inputs2.float())
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 在测试集上评估模型
        model.eval()  # 将模型设为评估模式
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for inputs1, inputs2, labels in test_loader:
                outputs = model(inputs1.float(), inputs2.float())
                predictions = outputs.squeeze().round().tolist()
                test_predictions.extend(predictions)
                test_labels.extend(labels.tolist())
        accuracy = accuracy_score(test_labels, test_predictions)
        recall = recall_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions)
        f1 = f1_score(test_labels, test_predictions)
        print(
            f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # 保存Precision最高的模型
        if precision > best_precision:
            best_precision = precision
            torch.save(model.state_dict(), "best_model11_precision_siamese_cnn_attention.pth")

        # 保存F1 score最高的模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model11_f1_siamese_cnn_attention.pth")

    # 输出最佳Precision和F1 Score
    print("Best Precision:", best_precision)
    print("Best F1 Score:", best_f1)
