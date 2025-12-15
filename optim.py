import math
import torch
from torch.optim import Optimizer

class Adam(Optimizer):
    """
    Cài đặt thuật toán Adam.
    Được đề xuất trong bài báo `Adam: A Method for Stochastic Optimization`_.
    """

    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8,
        weight_decay=0
    ):
        if not 0.0 <= lr:
            raise ValueError("Learning rate không hợp lệ: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Giá trị epsilon không hợp lệ: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Tham số beta tại index 0 không hợp lệ: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Tham số beta tại index 1 không hợp lệ: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Giá trị weight_decay không hợp lệ: {}".format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Thực hiện một bước tối ưu hóa (optimization step)."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam không hỗ trợ gradient thưa (sparse gradients), hãy cân nhắc sử dụng SparseAdam thay thế')

                state = self.state[p]

                # Khởi tạo trạng thái (State initialization)
                if len(state) == 0:
                    state['step'] = 0
                    # Trung bình động hàm mũ của gradient (Exponential moving average of gradient values)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Trung bình động hàm mũ của bình phương gradient (Exponential moving average of squared gradient values)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                # Cập nhật gradient nếu có weight_decay (trong phương pháp Adam gốc, weight decay được cộng trực tiếp vào gradient)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Phân rã moment thứ nhất và thứ hai (Decay the first and second moment running average coefficient)
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Tính toán mẫu số (denominator)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Tính toán bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Tính toán step size
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Cập nhật tham số
                # p_t = p_{t-1} - step_size * m_t / (sqrt(v_t) + eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class AdamW(Optimizer):
    """
    Cài đặt thuật toán AdamW.
    
    Thuật toán Adam gốc cập nhật tham số như sau:
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    Nếu sử dụng weight decay, gradient được sửa đổi:
    g = g + w * theta
    
    AdamW tách biệt weight decay khỏi cập nhật gradient (decoupling):
    theta = theta - lr * ( w * theta + m_hat / (sqrt(v_hat) + eps) )
    """
    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8,
        weight_decay=1e-2
    ):
        if not 0.0 <= lr:
            raise ValueError("Learning rate không hợp lệ: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Giá trị epsilon không hợp lệ: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Tham số beta tại index 0 không hợp lệ: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Tham số beta tại index 1 không hợp lệ: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Giá trị weight_decay không hợp lệ: {}".format(weight_decay))
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW không hỗ trợ gradient thưa (sparse gradients)')

                state = self.state[p]

                # Khởi tạo trạng thái (State initialization)
                if len(state) == 0:
                    state['step'] = 0
                    # Trung bình động hàm mũ của gradient
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Trung bình động hàm mũ của bình phương gradient
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Thực hiện phân rã trọng số (Perform stepweight decay)
                # Khác biệt chính của AdamW so với Adam nằm ở đây: weight decay được áp dụng trực tiếp lên dữ liệu
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Phân rã moment thứ nhất và thứ hai
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Cập nhật tham số
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
