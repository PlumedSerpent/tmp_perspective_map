import torch
from torch import nn
from torch.utils import model_zoo
import torch.nn.functional as F
import math
class HA(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels,out_channels, H_hat, norm_layer):
        super(HA, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.H_hat=H_hat
        inter_channels = int(in_channels/4)
        self.out_channels=out_channels
        self.g1 = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        position_1 = torch.arange(0, H_hat)
        self.register_buffer('position_1', position_1)
        div_term_1 = torch.exp(torch.arange(0, inter_channels, 2).float() * (-math.log(100.0) / inter_channels))
        self.register_buffer('div_term_1', div_term_1)
        self.g2 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels*2, 3, 1, 1, bias=False),
                                norm_layer(inter_channels*2))
        position_2 = torch.arange(0, H_hat)
        self.register_buffer('position_2', position_2)
        div_term_2 = torch.exp(torch.arange(0, inter_channels*2, 2).float() * (-math.log(100.0) / (inter_channels*2)))
        self.register_buffer('div_term_2', div_term_2)
        self.g3 = nn.Sequential(nn.Conv1d(inter_channels*2, self.out_channels, 3, 1, 1, bias=False),
                                norm_layer(self.out_channels))
        position_3 = torch.arange(0, H_hat)
        self.register_buffer('position_3', position_3)
        div_term_3 = torch.exp(torch.arange(0, self.out_channels, 2).float() * (-math.log(100.0) / (self.out_channels)))
        self.register_buffer('div_term_3', div_term_3)
        self.gamma = nn.Parameter(torch.zeros(1))
        # bilinear interpolate options

    def forward(self, x,pmap):
        b, c, h, w = x.size()
        pmap = F.interpolate(pmap,(h, w),mode="bilinear")
        x_h_low = F.interpolate(self.pool(x), (h, 1) ,mode="bilinear").view(b,c,h)
        pmap = F.interpolate(self.pool(pmap), (h, 1),mode="bilinear").squeeze(1).view(b,h,1)
        pmap = (pmap).long().float()
        Q1=F.relu_(self.g1(x_h_low))
        position_1=pmap
        div_term_1=self.div_term_1.unsqueeze(0)
        pe_1=torch.cat((torch.sin(position_1 * div_term_1),torch.cos(position_1 * div_term_1)),dim=-1)
        pe_1 = pe_1.transpose(1, 2)
        Q1=Q1+pe_1
        Q2=F.relu_(self.g2(Q1))
        position_2=pmap#self.position_2.unsqueeze(1).unsqueeze(0).expand(b,h//2,1)+(start_h.view(b,1,1).expand(b,h//2,1))//((400//h)*2)
        div_term_2=self.div_term_2.unsqueeze(0)
        pe_2=torch.cat((torch.sin(position_2 * div_term_2),torch.cos(position_2 *div_term_2)),dim=-1)
        pe_2 = pe_2.transpose(1, 2)
        Q2=Q2+pe_2
        Q3=F.sigmoid(self.g3(Q2))
        A= F.interpolate(Q3,h).view(b,self.out_channels,h,1)
        return A
# class HA(nn.Module):
#     """
#     Reference:
#     """
#     def __init__(self, in_channels,out_channels, H_hat, norm_layer):
#         super(HA, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d((None, 1))
#         self.H_hat=H_hat
#         inter_channels = int(in_channels/4)
#         self.out_channels=out_channels
#         self.g1 = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels))
#         position_1 = torch.arange(0, H_hat)
#         self.register_buffer('position_1', position_1)
#         div_term_1 = torch.exp(torch.arange(0, inter_channels, 2).float() * (-math.log(100.0) / inter_channels))
#         self.register_buffer('div_term_1', div_term_1)
#         self.g2 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels*2, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels*2))
#         position_2 = torch.arange(0, H_hat)
#         self.register_buffer('position_2', position_2)
#         div_term_2 = torch.exp(torch.arange(0, inter_channels*2, 2).float() * (-math.log(100.0) / (inter_channels*2)))
#         self.register_buffer('div_term_2', div_term_2)
#         self.g3 = nn.Sequential(nn.Conv1d(inter_channels*2, self.out_channels, 3, 1, 1, bias=False),
#                                 norm_layer(self.out_channels))
#         position_3 = torch.arange(0, H_hat)
#         self.register_buffer('position_3', position_3)
#         div_term_3 = torch.exp(torch.arange(0, self.out_channels, 2).float() * (-math.log(100.0) / (self.out_channels)))
#         self.register_buffer('div_term_3', div_term_3)
#         # bilinear interpolate options

#     def forward(self, x,start_h):
#         b, c, h, w = x.size()
#         x_h_low = F.interpolate(self.pool(x), (h//2, 1)).view(b,c,h//2)
#         Q1=F.relu_(self.g1(x_h_low))
#         position_1=self.position_1.unsqueeze(1).unsqueeze(0).expand(b,h//2,1)+(start_h.view(b,1,1).expand(b,h//2,1))//((400//h)*2)
#         div_term_1=self.div_term_1.unsqueeze(0)
#         pe_1=torch.cat((torch.sin(position_1 * div_term_1),torch.cos(position_1 * div_term_1)),dim=-1)
#         pe_1 = pe_1.transpose(1, 2)
#         Q1=Q1+pe_1
#         Q2=F.relu_(self.g2(Q1))
#         position_2=self.position_2.unsqueeze(1).unsqueeze(0).expand(b,h//2,1)+(start_h.view(b,1,1).expand(b,h//2,1))//((400//h)*2)
#         div_term_2=self.div_term_2.unsqueeze(0)
#         pe_2=torch.cat((torch.sin(position_2 * div_term_2),torch.cos(position_2 *div_term_2)),dim=-1)
#         pe_2 = pe_2.transpose(1, 2)
#         Q2=Q2+pe_2
#         Q3=F.sigmoid(self.g3(Q2))
#         A= F.interpolate(Q3,h).view(b,self.out_channels,h,1)
#         return A
class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.amp = BackEnd_amap()
        self.dmp = BackEnd_dmap()

        self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input,start_h):
        input = self.vgg(input)
        amp_out = self.amp(*input)
        dmp_out = self.dmp(start_h,*input)

        amp_out = self.conv_att(amp_out)
        dmp_out = amp_out * dmp_out
        dmp_out = self.conv_out(dmp_out)


        return dmp_out, amp_out

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3, conv5_3


class BackEnd_dmap(nn.Module):
    def __init__(self):
        super(BackEnd_dmap, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.strip_1 = HA(256, 256,25,nn.BatchNorm1d)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.strip_2 = HA(128, 128,50,nn.BatchNorm1d)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.strip_3 = HA(64, 64,100,nn.BatchNorm1d)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.strip_4 = HA(64, 32,100,nn.BatchNorm1d)

    def forward(self,start_h, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.upsample(conv5_3)

        input = torch.cat([input, conv4_3], 1)
        
        input = self.conv1(input)
        A = self.strip_1(input,start_h)
        input = self.conv2(input)
        input= input*A
        input = self.upsample(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        A = self.strip_2(input,start_h)
        input = self.conv4(input)
        input = input*A
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        A = self.strip_3(input,start_h)
        input = self.conv6(input)
        input = input*A
        
        A = self.strip_4(input,start_h)
        input = self.conv7(input)
        input = input*A
        return input
class BackEnd_amap(nn.Module):
    def __init__(self):
        super(BackEnd_amap, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.upsample(conv5_3)

        input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.upsample(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


if __name__ == '__main__':
    input = torch.randn(8, 3, 400, 400).cuda()
    model = Model().cuda()
    output, attention = model(input)
    print(input.size())
    print(output.size())
    print(attention.size())
