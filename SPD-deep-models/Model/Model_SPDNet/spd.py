import sys
sys.path.append('/home/zju/Python_Scripts/Riemannian_RNN/EcoGLibrary/SEED')

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Function
import numpy as np

from Model.Model_SPDNet.utils import *
from Model.Model_SPDNet import StiefelParameter

"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDTransform, self).__init__()
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        if self.increase_dim:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1,2), torch.bmm(output, weight))

        return output


class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = np.asarray([0] * input_size + [1] * (output_size-input_size), dtype=np.float32)
        self.register_buffer('add', torch.from_numpy(np.diag(add)))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)

        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1,2)))

        return output

"""
Yu, K., & Salzmann, M. (2017). Second-order convolutional neural networks. arXiv preprint arXiv:1703.06817.
"""
class ParametricVectorize(nn.Module):

    def __init__(self, input_size, output_size):
        super(ParametricVectorize, self).__init__()
        self.weight = nn.Parameter(torch.ones(output_size, input_size), requires_grad=True)

    def forward(self, input):
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight, input)
        output = torch.bmm(output, weight.transpose(1,2))
        output = torch.mean(output, 2)
        return output

"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDVectorize(nn.Module):

    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class SPDUnVectorizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        n = int(-.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
        output = input.new(len(input), n, n)
        output.fill_(0)
        mask_upper = np.triu_indices(n)
        mask_diag = np.diag_indices(n)
        for k, x in enumerate(input):
            output[k][mask_upper] = x
            output[k] = output[k] + output[k].t()   
            output[k][mask_diag] /= 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            n = int(-.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
            grad_input = input.new(len(input), input.size(1))
            mask = np.triu_indices(n)
            for k, g in enumerate(grad_output):
                grad_input[k] = g[mask]

        return grad_input


class SPDUnVectorize(nn.Module):

    def __init__(self):
        super(SPDUnVectorize, self).__init__()

    def forward(self, input):
        return SPDUnVectorizeFunction.apply(input)


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()
                
                g = symmetric(g)
                
                s_log_diag = s.log().diag()
                s_inv_diag = (1/s).diag()
                
                dLdV = 2*(g.mm(u.mm(s_log_diag)))
                dLdS = eye * (s_inv_diag.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())


        return grad_input


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDTangentSpace(nn.Module):

    def __init__(self, input_size, vectorize=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(input_size)

    def forward(self, input):
        output = SPDTangentSpaceFunction.apply(input)
        if self.vectorize:
            output = self.vec(output)

        return output


class SPDUnTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.exp_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                g = symmetric(g)
                
                s_exp_diag = s.exp().diag()
                
                dLdV = 2*(g.mm(u.mm(s_exp_diag)))
                dLdS = eye * (s_exp_diag.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())


        return grad_input


class SPDUnTangentSpace(nn.Module):

    def __init__(self, unvectorize=True):
        super(SPDUnTangentSpace, self).__init__()
        self.unvectorize = unvectorize
        if unvectorize:
            self.unvec = SPDUnVectorize()

    def forward(self, input):
        if self.unvectorize:
            input = self.unvec(input)
        output = SPDUnTangentSpaceFunction.apply(input)
        return output


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDRectifiedFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue

                g = symmetric(g)

                x = input[k]
                u, s, v = x.svd()
                
                max_mask = s > epsilon
                s_max_diag = s.clone(); s_max_diag[~max_mask] = epsilon; s_max_diag = s_max_diag.diag()
                Q = max_mask.float().diag()
                
                dLdV = 2*(g.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())
            
        return grad_input, None


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

    def forward(self, input):
        output = SPDRectifiedFunction.apply(input, self.epsilon)
        return output


class SPDPowerFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s = torch.exp(weight * torch.log(s))
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = None
        grad_weight = None

        eye = input.new(input.size(1))
        eye.fill_(1); eye = eye.diag()
        grad_input = input.new(input.size(0), input.size(1), input.size(2))
        grad_weight = weight.new(input.size(0), weight.size(0))
        for k, g in enumerate(grad_output):
            if len(g.shape) == 1:
                continue

            x = input[k]
            u, s, v = x.svd() 

            g = symmetric(g)
            
            s_log = torch.log(s)
            s_power = torch.exp(weight * s_log)

            s_power = s_power.diag()
            s_power_w_1 = weight * torch.exp((weight-1) * s_log)
            s_power_w_1 = s_power_w_1.diag()
            s_log = s_log.diag()
            
            grad_w = s_log.mm(u.t().mm(s_power.mm(u))).mm(g)
            grad_weight[k] = grad_w.diag()

            dLdV = 2*(g.mm(u.mm(s_power)))
            dLdS = eye * (s_power_w_1.mm(u.t().mm(g.mm(u))))
            
            P = s.unsqueeze(1)
            P = P.expand(-1, P.size(0))
            P = P - P.t()
            mask_zero = torch.abs(P) == 0
            P = 1 / P
            P[mask_zero] = 0            
            
            grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())

        grad_weight = grad_weight.mean(0)
        
        return grad_input, grad_weight


class SPDPower(nn.Module):

    def __init__(self, input_dim):
        super(SPDPower, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        # nn.init.normal(self.weight)

    def forward(self, input):
        output = SPDPowerFunction.apply(input, self.weight)
        return output
    

def Chol_de(A, n):
        L = A
        result = L[:, 0:1, 0:1]
        for i in range(1, n):
            j = i
            result = torch.cat([result, L[:, i:i + 1, :j + 1]], dim=2)
        result = result.reshape((-1, n * (n + 1) // 2))
        return result


class SPDNet(nn.Module):
    def __init__(self, input_size):
        super(SPDNet, self).__init__()
        self.input_size = input_size
        self.unit1 = input_size // 2
        self.trans1 = SPDTransform(input_size, self.unit1)
        self.rect1  = SPDRectified()
        self.tangent = SPDTangentSpace(self.unit1)
        self.linear_1 = nn.Linear(self.unit1 * (self.unit1 + 1) // 2, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = self.tangent(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        return x
    

class SPDNet_Chol(nn.Module):
    def __init__(self, input_size):
        super(SPDNet_Chol, self).__init__()
        self.input_size = input_size
        self.unit1 = input_size // 2
        self.trans1 = SPDTransform(input_size, self.unit1)
        self.rect1  = SPDRectified()
        self.linear_1 = nn.Linear(self.unit1 * (self.unit1 + 1) // 2, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = Chol_de(x, self.unit1)
        x = self.dropout(x)
        x = self.linear_1(x)
        return x


class MySPDNet(nn.Module):
    def __init__(self, n_class, input_size):
        super(MySPDNet, self).__init__()
        self.unit1 = input_size // 2
        self.unit_tangent = self.unit1 * (self.unit1 + 1) // 2
        self.trans1 = SPDTransform(input_size, self.unit1)
        self.rect1  = SPDRectified()
        self.tangent = SPDTangentSpace(self.unit1)
        self.cls = nn.Linear(self.unit_tangent, n_class)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = self.tangent(x)
        x = self.dropout(x)
        out = self.cls(x)
        
        return out


class MyLinearNet(nn.Module):
    def __init__(self, n_class, input_size):
        super(MyLinearNet, self).__init__()
        # self.tangent = SPDTangentSpace(self.unit2)
        self.input_size = input_size
        self.unit_tangent = input_size * (input_size + 1) // 2
        self.cls = nn.Linear(self.unit_tangent, n_class)
        self.dropout = nn.Dropout()

    def forward(self, x):
        rows, cols = torch.tril_indices(self.input_size, self.input_size)
        x = x[:, rows, cols]
        #x = self.tangent(x)
        x = self.dropout(x)
        out = self.cls(x)
        
        return out


class USPDNet_encoder(nn.Module):
    def __init__(self, input_size):
        super(USPDNet_encoder, self).__init__()
        self.unit1 = input_size - 9
        self.unit2 = self.unit1 - 9
        self.unit3 = self.unit2 - 9
        self.trans1 = SPDTransform(input_size, self.unit1)
        self.rect1  = SPDRectified()
        self.trans2 = SPDTransform(self.unit1, self.unit2)
        self.rect2  = SPDRectified()
        self.trans3 = SPDTransform(self.unit2, self.unit3)

    def forward(self, x):
        x_unit1 = self.trans1(x)
        x_unit1 = self.rect1(x_unit1)
        x_unit2 = self.trans2(x_unit1)
        x_unit2 = self.rect2(x_unit2)
        x_unit3 = self.trans3(x_unit2)
        
        return x_unit1, x_unit2, x_unit3


class USPDNet_decoder(nn.Module):
    def __init__(self, input_size):
        super(USPDNet_decoder, self).__init__()
        self.unit1 = input_size + 9
        self.unit2 = self.unit1 + 9
        self.unit3 = self.unit2 + 9

        self.trans1 = SPDTransform(input_size, self.unit1)
        self.rect1  = SPDRectified()
        self.trans2 = SPDTransform(self.unit1, self.unit2)
        self.rect2  = SPDRectified()
        self.trans3 = SPDTransform(self.unit2, self.unit3)

        self.tangent1 = SPDTangentSpace(self.unit1)
        self.tangent2 = SPDTangentSpace(self.unit2)
        self.untangent1 = SPDUnTangentSpace()
        self.untangent2 = SPDUnTangentSpace()


    def forward(self, x_unit1, x_unit2, x_unit3):
        x_de1 = self.trans1(x_unit3)
        x_de1 = self.rect1(x_de1)
        x_de1_tan = self.tangent1(x_de1)
        x_unit2_tan = self.tangent1(x_unit2)
        x_de1 = (x_de1_tan + x_unit2_tan) / 2
        x_de1 = self.untangent1(x_de1)
        #x_de1 += x_unit2

        x_de2 = self.trans2(x_de1)
        x_de2 = self.rect2(x_de2)
        x_de2_tan = self.tangent2(x_de2)
        x_unit1_tan = self.tangent2(x_unit1)
        x_de2 = (x_de2_tan + x_unit1_tan) / 2
        x_de2 = self.untangent2(x_de2)
        #x_de2 += x_unit1

        x_rec = self.trans3(x_de2)
        
        return x_rec


class USPDNet_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(USPDNet_AE, self).__init__()
        self.encoder = USPDNet_encoder(input_size)
        self.decoder = USPDNet_decoder(hidden_size)

    def forward(self, input):
        x_unit1, x_unit2, x_unit3 = self.encoder(input)
        x_rec = self.decoder(x_unit1, x_unit2, x_unit3)

        return x_rec
    

class USPDNet(nn.Module):
    def __init__(self, n_class, input_size, hidden_size, pretrain=None):
        super(USPDNet, self).__init__()

        if pretrain is None:
            self.encoder = USPDNet_encoder(input_size)
        else:
            self.encoder = pretrain

        self.tangent_size = hidden_size * (hidden_size + 1) // 2
        self.tangent = SPDTangentSpace(hidden_size)
        self.cls = nn.Linear(self.tangent_size, n_class)


    def forward(self, input):
        _, _, x_unit3 = self.encoder(input)

        x = self.tangent(x_unit3)
        out = self.cls(x)
        
        return out


class DreamNet(nn.Module):
    def __init__(self, n_class, input_size, hidden_size, device=None):
        super(DreamNet, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.num_layers = 4
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = nn.ModuleList([
                SPDTransform(input_size, hidden_size),
                SPDRectified(),
                SPDTransform(hidden_size, input_size),
                SPDRectified()
            ])
            self.blocks.append(block)

        self.tangent_size = hidden_size * (hidden_size + 1) // 2
        self.linearblocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = nn.ModuleList([
                SPDTangentSpace(hidden_size),
                nn.Linear(self.tangent_size, n_class)
            ])
            self.linearblocks.append(block)


    def forward(self, x):
        saved_outputs = {}
        out = torch.zeros(x.shape[0], self.n_class).to(self.device)
        hidden_last = torch.zeros(self.hidden_size, self.hidden_size).to(self.device)
        for idx, block in enumerate(self.blocks):
            for i, layer in enumerate(block):
                x = layer(x)
                if isinstance(layer, SPDTransform) and i == 0:
                    x += hidden_last
                    x_l = x.clone()
                    linearblock = self.linearblocks[idx]
                    for linearlayer in linearblock:
                        x_l = linearlayer(x_l)
                    out += x_l
                if isinstance(layer, SPDRectified) and i == 1:
                    hidden_last = x.clone()

        return out, x

