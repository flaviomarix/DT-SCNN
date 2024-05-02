from unicodedata import name
import torchvision
import torchvision.transforms as transforms
import os, time
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
names = 'STA_DT_1from5'
data_path = '.'  # input your path

device = torch.device("cuda")
# Hyper parameters
thresh = 0.5
lens = 1.5
probs = 0.5
alpha = 0.85
decay = 0.5
batch_size = 1  # increasing batch_size windows can help performance
num_epochs = 400
learning_rate = 0.001 
time_window = 1  # pre-train stage:5, re-train stage:1 


class ActFun(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1).float()

    @staticmethod
    def backward(ctx, grad_output):
        # au function
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - 1) < lens
        return grad_input * temp.float()


class ActFun_conv(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1).float()

    @staticmethod
    def backward(ctx, grad_output):
        # au function
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - 1) < lens
        return grad_input * temp.float() 


# membrane potential update
def mem_update(conv, x, mem, spike, Vth):
    mem = mem * decay * (1. - spike) + conv(x)
    spike = act_fun(mem / Vth)
    return mem, spike


def mem_update_conv(conv, x, mem, spike, Vth):
    mem = mem * decay * (1. - spike) + conv(x)
    spike = act_fun_conv(mem / Vth)
    return mem, spike


def dt_mem_update1(mem, Vth):
    spike = act_fun(mem / Vth)
    return spike


def out_mem_update(conv, x, mem):
    mem = mem + conv(x)
    return mem


act_fun = ActFun.apply
act_fun_conv = ActFun_conv.apply
# cnn layer :(in_plane,out_plane, stride,padding, kernel_size)
cfg_cnn = [(3, 48, 1, 1, 3),

           (96, 128, 1, 1, 3),
           # ap2

           (256, 192, 1, 1, 3),
           # ap2

           (384, 192, 1, 1, 3),

           (384, 128, 1, 1, 3),
           # ap2

           ]

cfg_kernel = [32, 32, 16, 8, 8, 4]
# fc layer
cfg_fc = [1024, 1024, 10]

# voting matrix
weights = torch.zeros(cfg_fc[-1], 10, device=device, requires_grad=False)  # cfg_fc[-1]
vote_num = cfg_fc[-1] // 10
for i in range(cfg_fc[-1]):
    weights.data[i][i // vote_num] = 10 / cfg_fc[-1]


def assign_optimizer(model, lrs=1e-3):
    rate = 1e-1
    vth_params = list(map(id, model.threshold.parameters()))
    # fc1_params = list(map(id, model.fc1.parameters()))
    # fc2_params = list(map(id, model.fc2.parameters()))
    # fc3_params = list(map(id, model.fc3.parameters()))
    base_params = filter(lambda p: id(p) not in vth_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        # {'params': model.fc1.parameters(), 'lr': lrs * rate},
        # {'params': model.fc2.parameters(), 'lr': lrs * rate},
        {'params': model.threshold.parameters(), 'lr': lrs * rate}, ]
        , lr=lrs)
    print('successfully reset lr')
    return optimizer


# def assign_optimizer(model, lrs=1e-3):
#     rate = 1e-1
#     optimizer = torch.optim.SGD(model.parameters(), lr=lrs, momentum=0.9)
#     print('successfully reset lr')
#     return optimizer

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.fc1 = nn.Linear(2 * cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0], bias=False)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=False)
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=False)
        # self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=False)
        self.softmax = nn.Softmax(dim=1)

        # Dual-threshold
        threshold = {}
        for l in range(7):
            threshold['t' + str(l + 1)] = nn.Parameter(torch.tensor(thresh))

        for l in range(5):
            threshold['g' + str(l + 1)] = nn.Parameter(torch.tensor(thresh))

        self.threshold = nn.ParameterDict(threshold)

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        # c1_mem_dt2 = c1_spike_dt2 = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        # c1_mem_dt3 = c1_spike_dt3 = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
        c5_mem = c5_spike = torch.zeros(batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)

        for step in range(time_window):
            # x = input > torch.rand(input.size(), device=device)
            x = input

            c1_mem, c1_spike = mem_update_conv(self.conv1, x.float(), c1_mem, c1_spike,
                                               Vth=getattr(self.threshold, 't' + str(1)))
            c1_spike_dt1 = dt_mem_update1(c1_mem, Vth=0.74*getattr(self.threshold, 'g' + str(1)))
            x = torch.cat((c1_spike, c1_spike_dt1), dim=1)

            c2_mem, c2_spike = mem_update_conv(self.conv2, F.dropout(x, p=probs, training=self.training), c2_mem,
                                               c2_spike, Vth=getattr(self.threshold, 't' + str(2)))
            c2_spike_dt1 = dt_mem_update1(c2_mem, Vth=getattr(self.threshold, 'g' + str(2)))
            x = torch.cat((c2_spike, c2_spike_dt1), dim=1)

            x = F.avg_pool2d(x, 2)
            x = F.dropout(x, p=probs, training=self.training)

            c3_mem, c3_spike = mem_update_conv(self.conv3, x, c3_mem, c3_spike,
                                               Vth=getattr(self.threshold, 't' + str(3)))
            c3_spike_dt1 = dt_mem_update1(c3_mem, Vth=getattr(self.threshold, 'g' + str(3)))
            x = torch.cat((c3_spike, c3_spike_dt1), dim=1)

            x = F.avg_pool2d(x, 2)
            x = F.dropout(x, p=probs, training=self.training)

            c4_mem, c4_spike = mem_update_conv(self.conv4, x, c4_mem, c4_spike,
                                               Vth=getattr(self.threshold, 't' + str(4)))
            c4_spike_dt1 = dt_mem_update1(c4_mem, Vth=getattr(self.threshold, 'g' + str(4)))
            x = torch.cat((c4_spike, c4_spike_dt1), dim=1)
            x = F.dropout(x, p=probs, training=self.training)

            c5_mem, c5_spike = mem_update_conv(self.conv5, x, c5_mem, c5_spike,
                                               Vth=getattr(self.threshold, 't' + str(5)))
            c5_spike_dt1 = dt_mem_update1(c5_mem, Vth=getattr(self.threshold, 'g' + str(5)))
            x = torch.cat((c5_spike, c5_spike_dt1), dim=1)
            x = F.avg_pool2d(x, 2)

            # x = F.max_pool2d(c3_spike, 2)
            # x = F.dropout(x, p=probs, training=self.training)

            # c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)

            # x = F.dropout(c4_spike, p=probs, training=self.training)
            # c5_mem, c5_spike = mem_update(self.conv5, x, c5_mem, c5_spike)
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, F.dropout(x, p=probs, training=self.training), h1_mem, h1_spike,
                                          Vth=getattr(self.threshold, 't' + str(6)))

            h2_mem, h2_spike = mem_update(self.fc2, F.dropout(h1_spike, p=probs, training=self.training), h2_mem,
                                          h2_spike, Vth=getattr(self.threshold, 't' + str(7)))

            h3_mem = out_mem_update(self.fc3, h2_spike, h3_mem)

            # c1_spike_np = c1_spike.cpu().numpy()
            # c1_spike_fkg_np = c1_spike_dt1.cpu().numpy()

            # # 将96个通道的特征图排列成一张图片
            # a = c1_spike[0]
            # a = a.unsqueeze(1)
            # feature_map = torchvision.utils.make_grid(a, nrow=8)

            # # 显示图片
            # plt.imshow(feature_map.permute(1, 2, 0).cpu())
            # plt.show()
            # a = c1_spike_dt1[0]
            # a = a.unsqueeze(1)
            # feature_map = torchvision.utils.make_grid(a, nrow=8)

            # # 显示图片
            # plt.imshow(feature_map.permute(1, 2, 0).cpu())
            # plt.show()

        # outputs = self.softmax(h3_mem)
        outputs = h3_mem
        # print(weights)

        return outputs


# Data preprocessing
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Grayscale()

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Grayscale()
])

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

net = SCNN()
net = net.to(device)

# criterion = nn.MSELoss() # Mean square error loss
criterion = nn.CrossEntropyLoss()
best_acc = 0  # best test accuracy
best_net = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
optimizer = assign_optimizer(net, lrs=learning_rate)


# using SGD+CosineAnnealing could achieve better results
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training
def train(epoch):
    # for k,v in net.named_parameters():
    #     if 'conv1' in k:
    #         v.requires_grad = False
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # starts = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        # loss = criterion(outputs.cpu(), labels_)
        loss = criterion(outputs.cpu(), targets)
        loss.backward()
        optimizer.step()
        # print('outputs',outputs)
        # print('labels_',labels_)
        # tg = 'conv1'
        # for name, parms in net.named_parameters():
        #     # if tg in name:
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad.abs().max())
        #     # print("=== ???")

        train_loss += loss.item()
        _, predicted = outputs.cpu().max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        if batch_idx % 50 == 0:
            elapsed = time.time() - starts
            print(batch_idx, 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # for key, value in sorted(net.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
            #     temp1 = temp1+[round(value.item(),5)]
            # print(net.threshold.items())
    print('Time past: ', elapsed, 's', 'Iter number:', epoch)
    loss_train_record.append(train_loss)


def test(epoch):
    global best_acc
    global best_net
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            # labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            # loss = criterion(outputs.cpu(), labels_)
            loss = criterion(outputs.cpu(), targets)
            test_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print(net.threshold.items())
        loss_test_record.append(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    acc_record.append(acc)

    if best_acc < acc:
        best_acc = acc
        best_net = net.state_dict()
        print('Saving..')
    state = {
        'net': net.state_dict(),
        'best_net': best_net,
        'best_acc': best_acc,
        'acc': acc,
        'epoch': epoch,
        'acc_record': acc_record,
        'loss_train_record': loss_train_record,
        'loss_test_record': loss_test_record,
    }
    if not os.path.isdir('./Vout'):
        os.mkdir('./Vout')
    torch.save(state, './Vout/' + names + '.t7')


def pure_test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(targets)

            # labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            # loss = criterion(outputs.cpu(), labels_)
            loss = criterion(outputs.cpu(), targets)
            test_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        loss_test_record.append(test_loss)


import cv2
import torch
def fmap_show_test():
    global best_acc
    net.eval()
    with torch.no_grad():

        # 读取PNG彩色图片
        img = cv2.imread('cat.png', cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 100.0

        # # 调整尺寸为（1，28，28，3）
        # img_tensor = torch.nn.functional.interpolate(img_tensor, size=(28, 28), mode='bilinear', align_corners=False)

        print(img_tensor.shape)

        inputs = img_tensor.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        print(outputs)



# Load pre trained model
# state = torch.load('./checkpoint/spiking_cnn_model.t7')
# state = torch.load('./Vout/STA_FKG_noDiv_5.t7')
state = torch.load('./Vout/'+names+'.t7')
net.load_state_dict(state['best_net'])
print(state['best_acc'])

for epoch in range(start_epoch, start_epoch+num_epochs):

    starts = time.time()
    train(epoch)
    test(epoch)
    elapsed =  time.time() - starts
    optimizer = lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=100)
    print (" \n\n\n\n")
    print('Time past: ',elapsed,'s', 'Iter number:', epoch)
    print(names)
    print(best_acc)

# fmap_show_test()
print(best_acc)
