from torch import nn
class GestureMLP(nn.Module):
    def __init__(self,input_size=42,hidden_size=128,num_classes=6):
        super(GestureMLP,self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc2= nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x