import sys
sys.path.append('/home/zju/Python_Scripts/Riemannian_RNN/EcoGLibrary/SEED')


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from Model.Model_SPDNet.spd import SPDTangentSpace, SPDUnTangentSpace

class SPDGeometricAttention(nn.Module):
    def __init__(self, in_channels):
        """流形几何注意力层
        Args:
            in_channels: 输入通道数
        """
        super().__init__()
        self.attention = nn.Parameter(torch.ones(in_channels))
        
    def forward(self, X_list):
        # X_list: List[[batch, n, n]] 输入SPD矩阵列表
        # 应用通道注意力权重
        weighted = [a * X for a, X in zip(F.softmax(self.attention, dim=0), X_list)]
        return weighted

class SPDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, epsilon=1e-6):
        """多通道SPD卷积层
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核尺寸k
            epsilon: 正定修正系数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.epsilon = epsilon
        
        # 每个输出通道对应in_channels个卷积核
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, self.k, self.k)*0.01)
            for _ in range(out_channels)
        ])
        
    def _compute_W(self, V):
        """计算单个SPD卷积核 W = V^T V + εI"""
        return torch.matmul(V.transpose(-1,-2), V) + self.epsilon * torch.eye(self.k, device=V.device)
    
    def forward(self, X_list):
        """输入: List[[batch, n, n]] SPD矩阵列表
           输出: List[[batch, m, m]] SPD矩阵列表 (m = n - k + 1)
        """
        batch_size, n, _ = X_list[0].shape
        m = n - self.k + 1
        
        # 为每个输出通道计算结果
        output_channels = []
        for V in self.V:  # 遍历每个输出通道的卷积核组
            # 对每个输入通道进行卷积
            channel_results = []
            for X, v in zip(X_list, V):  # X:[batch,n,n], v:[k,k]
                W = self._compute_W(v)
                conv_result = torch.zeros((batch_size, m, m), device=X.device)
                
                # 滑动窗口卷积
                for i in range(m):
                    for j in range(m):
                        block = X[:, i:i+self.k, j:j+self.k]
                        conv_result[:,i,j] = torch.einsum('bij,ij->b', block, W)
                
                channel_results.append(conv_result)
            
            # 跨输入通道求和得到单个输出通道
            output_channel = torch.sum(torch.stack(channel_results), dim=0)
            output_channel = F.relu(output_channel / torch.norm(output_channel, p='fro'))
            output_channels.append(output_channel)
        
        return output_channels

class SPDManifoldPooling(nn.Module):
    def __init__(self, input_size, pool_size=2):
        """流形平均池化层
        Args:
            pool_size: 池化下采样倍数
        """
        super().__init__()
        self.pool_size = pool_size
        self.tangent = SPDTangentSpace(input_size, vectorize=False)
        self.untangent = SPDUnTangentSpace(unvectorize=False)
        
    def _log_map(self, X):
        """将SPD矩阵映射到切空间(矩阵对数)"""
        return torch.matrix_log(X)
    
    def _exp_map(self, H):
        """将切空间矩阵映射回流形(矩阵指数)"""
        return torch.matrix_exp(H)
    
    def forward(self, X_list):
        """输入: List[[batch, n, n]] SPD矩阵列表
           输出: List[[batch, m, m]] SPD矩阵列表 (m = n//pool_size)
        """
        pooled_list = []
        for X in X_list:
            # Step 1: 映射到切空间
            # log_X = self._log_map(X)
            log_X = self.tangent(X)
            
            # Step 2: 经典平均池化
            batch, n, _ = log_X.shape
            m = n // self.pool_size
            pooled = F.avg_pool2d(log_X.unsqueeze(1), kernel_size=self.pool_size).squeeze(1)
            
            # Step 3: 映射回流形空间
            # exp_pooled = self._exp_map(pooled)
            exp_pooled = self.untangent(pooled)
            pooled_list.append(exp_pooled)
            
        return pooled_list

class SPDCNN(nn.Module):
    def __init__(self, in_size, in_channels, num_classes, channels=[4], kernel_sizes=[3], pool_size=2):
        """完整SPDCNN网络
        Args:
            in_size: 输入SPD矩阵尺寸n*n
            num_classes: 分类类别数
            channels: 各层通道数配置 [r(1), r(2),...]
            kernel_sizes: 各层卷积核尺寸
            pool_size: 池化下采样倍数
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.pool = SPDManifoldPooling(in_size, pool_size)
        
        # 构建网络层
        current_channels = in_channels  # 初始输入通道数
        for l in range(len(channels)):
            # 注意力层
            attn_layer = SPDGeometricAttention(current_channels)
            # 卷积层
            conv_layer = SPDConv2d(in_channels=current_channels,
                                 out_channels=channels[l],
                                 kernel_size=kernel_sizes[l])
            # 池化层
            self.layers.append(nn.ModuleList([attn_layer, conv_layer]))
            
            # 更新参数
            current_channels = channels[l]
            in_size = (in_size - kernel_sizes[l] + 1) // pool_size
            
        # 最终特征组装层
        self.final_size = in_size * (in_size + 1) // 2
        self.num_channels = channels[-1]
        
        # 分类层
        self.classifier = nn.Linear(self.num_channels * self.final_size, num_classes)
        
    def _assemble_spd(self, X_list):
        """组装块对角SPD矩阵 Z = diag(Y1,Y2,...)"""
        batch_size = X_list[0].shape[0]
        device = X_list[0].device
        
        # 计算总尺寸
        block_size = X_list[0].shape[-1]
        total_size = block_size * len(X_list)
        Z = torch.zeros(batch_size, total_size, total_size, device=device)
        
        # 填充块对角
        for i, X in enumerate(X_list):
            start = i * block_size
            end = (i+1) * block_size
            Z[:, start:end, start:end] = X
            
        return Z
    
    def _vectorize_spd(self, X_list):
        batch_size, output_size = X_list[0].shape[0], X_list[1].shape[1]
        block_size = output_size * (output_size + 1) // 2
        output_channels = len(X_list)
        rows, cols = torch.tril_indices(output_size, output_size)
        device = X_list[0].device

        Z = torch.zeros((batch_size, output_channels * block_size), device=device)
        for i, x in enumerate(X_list):
            x = x[:, rows, cols]
            Z[:, i * block_size: (i + 1) * block_size] = x

        return Z

        
    def forward(self, X):
        X_list = []
        input_channels = X.shape[1]
        for i in range(input_channels):
            X_list.append(X[:, i])
        
        # 通过各层处理
        for attn_layer, conv_layer in self.layers:
            # 应用注意力
            # weighted = attn_layer(X_list)
            weighted = X_list
            
            # 卷积处理
            conved = conv_layer(weighted)
            
            # 应用池化
            pooled = self.pool(conved)
            
            X_list = pooled
        
        # 组装最终SPD矩阵
        # Z = self._assemble_spd(X_list)
        Z = self._vectorize_spd(X_list)
        
        # 展平分类
        batch_size = Z.shape[0]
        return self.classifier(Z.view(batch_size, -1))


class SPDCNN_small(nn.Module):
    def __init__(self, in_size, in_channels, num_classes, channels=[5], kernel_sizes=[1], pool_size=1):
        """完整SPDCNN网络
        Args:
            in_size: 输入SPD矩阵尺寸n*n
            num_classes: 分类类别数
            channels: 各层通道数配置 [r(1), r(2),...]
            kernel_sizes: 各层卷积核尺寸
            pool_size: 池化下采样倍数
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.pool = SPDManifoldPooling(in_size, pool_size)
        
        # 构建网络层
        current_channels = in_channels  # 初始输入通道数
        for l in range(len(channels)):
            # 注意力层
            attn_layer = SPDGeometricAttention(current_channels)
            # 卷积层
            conv_layer = SPDConv2d(in_channels=current_channels,
                                 out_channels=channels[l],
                                 kernel_size=kernel_sizes[l])
            # 池化层
            self.layers.append(nn.ModuleList([attn_layer, conv_layer]))
            
            # 更新参数
            current_channels = channels[l]
            in_size = (in_size - kernel_sizes[l] + 1) // pool_size
            
        # 最终特征组装层
        self.final_size = in_size * (in_size + 1) // 2
        self.num_channels = channels[-1]
        
        # 分类层
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(self.num_channels * self.final_size, num_classes)
        
    def _assemble_spd(self, X_list):
        """组装块对角SPD矩阵 Z = diag(Y1,Y2,...)"""
        batch_size = X_list[0].shape[0]
        device = X_list[0].device
        
        # 计算总尺寸
        block_size = X_list[0].shape[-1]
        total_size = block_size * len(X_list)
        Z = torch.zeros(batch_size, total_size, total_size, device=device)
        
        # 填充块对角
        for i, X in enumerate(X_list):
            start = i * block_size
            end = (i+1) * block_size
            Z[:, start:end, start:end] = X
            
        return Z
    
    def _vectorize_spd(self, X_list):
        batch_size, output_size = X_list[0].shape[0], X_list[0].shape[1]
        block_size = output_size * (output_size + 1) // 2
        output_channels = len(X_list)
        rows, cols = torch.tril_indices(output_size, output_size)
        device = X_list[0].device

        Z = torch.zeros((batch_size, output_channels * block_size), device=device)
        for i, x in enumerate(X_list):
            x = x[:, rows, cols]
            Z[:, i * block_size: (i + 1) * block_size] = x

        return Z

        
    def forward(self, X):
        X_list = []
        input_channels = X.shape[1]
        for i in range(input_channels):
            X_list.append(X[:, i])
        
        # 通过各层处理
        for attn_layer, conv_layer in self.layers:
            # 应用注意力
            # weighted = attn_layer(X_list)
            weighted = X_list
            
            # 卷积处理
            # conved = conv_layer(weighted)
            conved = weighted
            
            # 应用池化
            # pooled = self.pool(conved)
            pooled = conved
            
            X_list = pooled
        
        # 组装最终SPD矩阵
        # Z = self._assemble_spd(X_list)
        Z = self._vectorize_spd(X_list)
        
        # 展平分类
        batch_size = Z.shape[0]
        return self.classifier(self.dropout(Z.view(batch_size, -1)))

