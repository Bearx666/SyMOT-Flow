import torch

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化阴影变量
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class TeacherStudentEMA:
    def __init__(self, model_tch, model_std, decay):
        self.model_tch = model_tch
        self.model_std = model_std
        self.decay = decay

    def update(self):
        for (_, param_tch), (_, param_std) in zip(self.model_tch.named_parameters(), self.model_std.named_parameters()):
            param_tch.data = self.decay * param_tch.data + (1 - self.decay) * param_std.data
    


if __name__ == '__main__':
    class myNet(torch.nn.Module):
        def __init__(self):
            super(myNet, self).__init__()
            self.fc = torch.nn.Linear(2, 3)
        def forward(self, x):
            return self.fc(x)
    def print_param(net):
        for param in net.parameters():
            print(param.data)
        
    net = myNet()
    x = torch.ones(5,2)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
    ema = EMA(net, 0.9)
    print_param(net)
    for _ in range(1):
        optimizer.zero_grad()
        z = net(x)
        loss = torch.square(z).mean()
        loss.backward()
        optimizer.step()
        print_param(net)
        ema.update()
    # print_param(net)
    ema.apply_shadow()
    print_param(net)
    ema.restore()
    print_param(net)