# 导入PyTorch的神经网络模块  
import torch.nn as nn  
# 导入PyTorch库  
import torch  
  
# 定义一个名为GAM_Attention的类，继承自nn.Module  
class GAM_Attention(nn.Module):  
    # 初始化函数，定义模型参数  
    def __init__(self, in_channels, out_channels, rate=4):  
        # 调用父类的初始化函数  
        super(GAM_Attention, self).__init__()  
  
        # 定义通道注意力模块  
        self.channel_attention = nn.Sequential(  
            # 线性层，将输入通道数减少到in_channels/rate，使用ReLU激活函数  
            nn.Linear(in_channels, int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            # 线性层，将通道数恢复到原始的in_channels  
            nn.Linear(int(in_channels / rate), in_channels)  
        )  
  
        # 定义空间注意力模块  
        self.spatial_attention = nn.Sequential(  
            # 卷积层，将输入通道数减少到in_channels/rate，卷积核大小为7x7，使用ReLU激活函数和批量归一化层  
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            # 卷积层，将通道数增加到out_channels，卷积核大小为7x7，使用批量归一化层  
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(out_channels)  
        )  
  
    # 前向传播函数，定义了数据通过模型的过程  
    def forward(self, x):  
        b, c, h, w = x.shape  # 获取输入x的形状（batch size, channels, height, width）  
        # 对输入x进行维度变换和展平操作，以便通过通道注意力模块  
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
        # 通过通道注意力模块，得到通道权重x_att_permute  
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  
        # 将通道权重x_att_permute转回原始维度顺序，并乘以输入x得到x_channel_att，使用sigmoid激活函数归一化权重值到[0,1]区间内    
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  
        x = x * x_channel_att  # 应用通道注意力权重  
  
        # 通过空间注意力模块，得到空间权重x_spatial_att，使用sigmoid激活函数归一化权重值到[0,1]区间内  
        x_spatial_att = self.spatial_attention(x).sigmoid()  
        out = x * x_spatial_att  # 应用空间注意力权重  
  
        return out  # 返回处理后的输出
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class C2fGAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GAM_Attention(self.c, self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
  
# 主函数入口，用于测试模型功能  
if __name__ == '__main__':  
    # 创建一个形状为(batch size=1, channels=64, height=32, width=32)的随机输入张量x  
    x = torch.randn(1, 64, 32, 32)  
    b, c, h, w = x.shape  # 获取输入x的形状（batch size, channels, height, width）  
    net = GAM_Attention(in_channels=c, out_channels=c)  # 创建一个通道数为64的GAM_Attention模型实例net  
    y = net(x)  # 将输入x通过模型net得到输出y
    print(y.shape)
