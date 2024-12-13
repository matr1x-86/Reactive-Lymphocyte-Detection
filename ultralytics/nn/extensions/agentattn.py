import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class AgentAttention(nn.Module):  # 定义一个名为AgentAttention的类，继承自PyTorch的nn.Module类，表示这是一个神经网络模块。
    # 定义类的构造函数，接收以下参数：  
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., shift_size=0, agent_num=49, **kwargs):
        super().__init__()  # 调用父类的构造函数，进行必要的初始化。  
        self.dim = dim  # 输入的维度。  
        self.idf = nn.Identity()  # 创建一个恒等映射，不做任何改变。  
        self.num_heads = num_heads  # 多头注意力的头数。  
        head_dim = dim // num_heads  # 每个头的维度。  
        self.scale = head_dim ** -0.5  # 注意力的缩放因子。  
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 定义一个线性层，用于产生query、key和value。  
        self.attn_drop = nn.Dropout(attn_drop)  # 定义注意力权重的dropout层。  
        self.proj = nn.Linear(dim, dim)  # 定义一个线性层，用于投影。  
        self.proj_drop = nn.Dropout(proj_drop)  # 定义输出投影的dropout层。  
        self.softmax = nn.Softmax(dim=-1)  # 定义softmax函数。  
        self.shift_size = shift_size  # 定义窗口的移动大小。  
        self.agent_num = agent_num  # 定义agent的数量。  
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1,
                             groups=dim)  # 定义一个2D卷积层。
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))  # 定义一个参数，表示attention的偏置。  
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        # 使用截断正态分布初始化an_bias参数，其中std表示标准差，这里设置为0.02
        trunc_normal_(self.an_bias, std=.02)
        # 使用截断正态分布初始化na_bias参数，同样设置标准差为0.02
        trunc_normal_(self.na_bias, std=.02)
        # 计算pool_size的值，它是agent_num的平方根的整数部分
        pool_size = int(agent_num ** 0.5)
        # 定义一个自适应平均池化层，其输出大小为(pool_size, pool_size)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x):
        # 获取输入x经过idf层的输出，即shortcut
        shortcut = self.idf(x)

        # 获取shortcut的形状，并分别提取高和宽
        h, w = shortcut.shape[2], shortcut.shape[3]

        # 将输入x展平，并交换维度2和1的位置
        x = x.flatten(2).transpose(1, 2)

        # 获取展平后x的形状，并分别提取batch_size、序列长度和通道数
        b, n, c = x.shape

        # 定义多头注意力中的头数
        num_heads = self.num_heads

        # 计算每个头的维度
        head_dim = c // num_heads

        # 对输入x进行qkv变换，并重新整形和转置
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)

        # 将qkv的结果分为q、k和v三个部分
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 输出q、k、v的形状信息
        # q, k, v: b, n, c

        # 对q进行池化操作，得到agent_tokens，并对形状进行转置和重新整形
        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        # 对q、k、v进行形状的重新整形和转置，以便进行多头注意力计算
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # 对agent_tokens进行形状的重新整形和转置，以便与q、k、v进行多头注意力计算
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # 初始化ah_bias和aw_bias，这两个偏置用于位置编码
        ah_bias = torch.as_tensor(torch.zeros(1, num_heads, self.agent_num, h, 1).to(x.device).detach(), dtype=x.dtype)
        aw_bias = torch.as_tensor(torch.zeros(1, num_heads, self.agent_num, 1, w).to(x.device).detach(), dtype=x.dtype)

        # 对an_bias进行上采样，得到与输入x相同的大小
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(h, w), mode='bilinear')
        # 对上采样后的位置偏置进行形状调整，使其与后面的计算相匹配
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # 将ah_bias和aw_bias拼接，并与上采样后的位置偏置相加
        position_bias2 = (ah_bias + aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        # 对agent_tokens进行softmax操作，并与key进行矩阵乘法，再加上位置偏置
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        # 应用dropout
        agent_attn = self.attn_drop(agent_attn)
        # 对value进行加权求和，得到agent_v
        agent_v = agent_attn @ v

        # 初始化ha_bias和wa_bias，这两个偏置用于另一个位置编码
        ha_bias = torch.as_tensor(torch.zeros(1, num_heads, h, 1, self.agent_num).to(x.device).detach(), dtype=x.dtype)
        wa_bias = torch.as_tensor(torch.zeros(1, num_heads, 1, w, self.agent_num).to(x.device).detach(), dtype=x.dtype)

        # 对na_bias进行上采样，得到与输入x相同的大小
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(h, w), mode='bilinear')
        # 对上采样后的位置偏置进行形状调整，使其与后面的计算相匹配
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        # 将ha_bias和wa_bias拼接，并与上采样后的位置偏置相加
        agent_bias2 = (ha_bias + wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        # 对q和agent_tokens进行softmax操作，并进行矩阵乘法，再加上位置偏置
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        # 应用dropout
        q_attn = self.attn_drop(q_attn)
        # 对value进行加权求和，得到x的一部分结果
        x = q_attn @ agent_v

        # 对x进行形状调整，使其与shortcut相匹配
        x = x.transpose(1, 2).reshape(b, n, c)
        # 对v进行形状调整，使其与shortcut相匹配
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 将x与shortcut相加，并应用dwc层（可能是某种变换或池化层）
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        # 通过一个线性层proj进行变换
        x = self.proj(x)

        # 通过一个dropout层proj_drop进行dropout操作，以防止过拟合
        x = self.proj_drop(x)

        # 将x的形状重新调整为(b, h, w, c)，其中b是batch size，h和w是高和宽，c是通道数
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # 将shortcut与sigmoid函数的结果相乘，sigmoid函数可以将输入映射到0到1之间，这里可能是为了将输出限制在一个合理的范围内
        x = shortcut * torch.sigmoid(x)
        return x

    # 定义一个extra_repr方法，该方法返回一个描述对象的字符串
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    # 定义一个flops方法，该方法接受一个参数N，表示输入序列的长度
    def flops(self, N):
        # 初始化浮点运算次数的计数器为0  
        flops = 0
        # 这行代码被注释掉了，它似乎是计算qkv矩阵的FLOPs，其中qkv是query、key和value矩阵  
        # qkv = self.qkv(x)  
        # flops += N * self.dim * 3 * self.dim  
        # 这行代码计算了qkv矩阵的FLOPs，其中N是序列长度，self.dim是维度大小，3是因为qkv有3个矩阵（query、key和value）  
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))  
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # 这行代码计算了查询矩阵（q）和键矩阵（k）转置的乘法操作的FLOPs，其中self.num_heads是头数，N是序列长度，self.dim // self.num_heads是每个头的维度大小  
        # x = (attn @ v)  
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # 这行代码计算了注意力权重矩阵（attn）和值矩阵（v）的乘法操作的FLOPs  
        # x = self.proj(x)  
        flops += N * self.dim * self.dim
        # 这行代码计算了经过线性变换后的输出的FLOPs，其中N是序列长度，self.dim是维度大小  
        return flops

if __name__ == '__main__':

    input = torch.randn(1, 128, 32, 32)
    m = AgentAttention(128, 8)
    output = m(input)
    print(output.shape)
    # print(input[..., ::2, ::2].shape)