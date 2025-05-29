import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert self.config.n_channels is not None
        assert self.config.n_filters is not None

        self.input_conv = nn.Conv2d(config.n_channels, config.n_filters, kernel_size = (3,3), stride = 1, padding=1)
        self.bn1 = nn.BatchNorm2d(config.n_filters)
        self.relu = nn.ReLU(inplace=True)

        self.residual_tower = nn.ModuleList([Block(config) for _ in range(config.n_BLOCKS)])
        self.value_head = ValueHead(config)
        self.policy_head = PolicyHead(config)
            
    def forward(self, input,  targets = None):
        x = self.input_conv(input)
        x = self.relu(self.bn1(x))
        for block in self.residual_tower:
            x = block(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
       
    
        if targets is not None:
            value_loss = F.mse_loss(value, targets['value'])
            policy_loss = F.cross_entropy(policy, targets['policy'])
            loss = value_loss + policy_loss
        else:
            loss = None

        return value, policy, loss

    def forward(self, input, targets=None):
        x = self.input_conv(input)
        x = self.relu(self.bn1(x))

        for block in self.residual_tower:
            x = block(x)
        
        value = self.value_head(x)
        policy = self.policy_head(x)

        if targets is not None:
            value_loss = F.mse_loss(value, targets['value'])
            policy_loss = F.cross_entropy(policy, targets['policy'])
            loss = value_loss + policy_loss
        else:
            loss = None
            policy_loss =None
            value_loss =None

        return value, policy, loss, policy_loss, value_loss





class ValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.value_conv1 = nn.Conv2d(config.n_filters, 32, kernel_size=(3,3), stride=1, padding=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_conv2 = nn.Conv2d(32, 1, kernel_size=(1,1), stride=1)
        self.value_fc1 = nn.Linear(8 * 8, 128, bias=True)
        self.value_fc2 = nn.Linear(128, 1, bias=True)
    
    def forward(self, x):
        value = self.value_conv1(x)
        value = self.value_conv2(self.value_bn(value))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        return self.value_fc2(value).squeeze(-1)


class PolicyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.conv1 = nn.Conv2d(config.n_filters, config.n_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(config.n_filters)
        self.conv2 = nn.Conv2d(config.n_filters, 76, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(76 * 8 * 8, 76 * 8 * 8, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.conv1 = nn.Conv2d(config.n_filters,config.n_filters,kernel_size = (3,3), stride = 1, padding=1)
        self.bn1 = nn.BatchNorm2d(config.n_filters)
        self.conv2 = nn.Conv2d(config.n_filters,config.n_filters,kernel_size = (3,3), stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(config.n_filters)
        self.selayer = SELayer(config.n_filters, config.SE_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x1=x
        x=self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        x = self.selayer(x)
        x = x+x1
        x= self.relu(x)
        return x


class SELayer(nn.Module):
    def __init__(self, n_filters, SE_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(n_filters, SE_channels, bias=True)
        self.fc2 = nn.Linear(SE_channels,  2 * n_filters, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.size()
        out = self.global_pool(x).view(B, C)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        P, Q = torch.split(out, C, dim = 1)
        P = P.view(B, C, 1, 1)
        Q = Q.view(B,C,1,1)
        Z = torch.sigmoid(P)
        return Z * x + Q
    
@dataclass
class ModelConfig:
    n_channels = 22
    n_filters = 256
    n_BLOCKS = 20
    SE_channels = 32
    policy_channels = 76

























#<=====================================================Unit tests=====================================================>

def test_model_forward_pass():
    config = ModelConfig()
    config.n_channels = 22
    config.n_filters = 128
    config.n_BLOCKS = 3
    config.SE_channels = 16
    config.policy_channels = 76
    model = NeuralNetwork(config)
    batch_size = 2
    input_tensor = torch.randn(batch_size, config.n_channels, 8, 8)
    
    value, policy, loss, policy_loss, value_loss = model(input_tensor)
    assert value.shape == (batch_size,), f"Value shape should be ({batch_size},), got {value.shape}"
    assert policy.shape == (batch_size, 76*8*8), f"Policy shape should be ({batch_size}, {76*8*8}), got {policy.shape}"
    assert loss is None, "Loss should be None when no targets provided"
    targets = {
        'value': torch.randn(batch_size),
        'policy': torch.randint(0, 76*8*8, (batch_size,))
    }
    value, policy, loss, policy_loss, value_loss = model(input_tensor, targets)
    print(f'value: {value}')
    print(f'policy: {policy.shape}')
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor when targets are provided"
    assert loss.item() > 0, "Loss should be positive"
    print("Forward pass test passed!")

def test_value_head():
    config = ModelConfig()
    value_head = ValueHead(config)
    batch_size = 3
    input_tensor = torch.randn(batch_size, config.n_filters, 8, 8)
    output = value_head(input_tensor)
    assert output.shape == (batch_size,), f"Value head output shape should be ({batch_size},), got {output.shape}"
    print("Value head test passed!")

def test_policy_head():
    config = ModelConfig()
    policy_head = PolicyHead(config)
    batch_size = 4
    input_tensor = torch.randn(batch_size, config.n_filters, 8, 8)
    output = policy_head(input_tensor)
    assert output.shape == (batch_size, 76*8*8), f"Policy head output shape should be ({batch_size}, {76*8*8}), got {output.shape}"
    print("Policy head test passed!")

def test_residual_block():
    config = ModelConfig()
    block = Block(config)
    batch_size = 2
    input_tensor = torch.randn(batch_size, config.n_filters, 8, 8)
    output = block(input_tensor)
    assert output.shape == input_tensor.shape, f"Block should maintain shape, got {output.shape} vs {input_tensor.shape}"
    print("Residual block test passed!")

def test_se_layer():
    n_filters = 64
    se_channels = 32
    se_layer = SELayer(n_filters, se_channels)
    batch_size = 3
    input_tensor = torch.randn(batch_size, n_filters, 8, 8)
    output = se_layer(input_tensor)
    assert output.shape == input_tensor.shape, f"SELayer should maintain shape, got {output.shape} vs {input_tensor.shape}"
    print("SELayer test passed!")




if __name__ == "__main__":
    test_value_head()
    test_policy_head()
    test_residual_block()
    test_se_layer()
    test_model_forward_pass()
    print("All tests passed!")