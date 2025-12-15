import torch

class LRScheduler:
    """
    Learning Rate Scheduler cho mô hình Transformer.
    
    Lớp này điều chỉnh learning rate theo số bước huấn luyện (steps) dựa trên công thức
    được đề xuất trong bài báo "Attention Is All You Need".
    
    Công thức:
    lrate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
    
    Điều này tương ứng với việc tăng learning rate tuyến tính trong giai đoạn warmup đầu tiên,
    và sau đó giảm nó tỷ lệ với nghị đảo căn bậc hai của số bước (inverse square root of the step number).
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        """
        Khởi tạo Scheduler.
        
        Args:
            optimizer: Optimizer cần điều chỉnh learning rate (ví dụ: Adam).
            d_model: Kích thước của vector đặc trưng (dimension of model).
            warmup_steps: Số bước warmup.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        """
        Cập nhật learning rate sau mỗi bước huấn luyện.
        
        Phương thức này nên được gọi sau mỗi lần optimizer.step().
        """
        self.step_num += 1
        
        # Tính toán learning rate mới
        arg1 = self.step_num ** -0.5
        arg2 = self.step_num * (self.warmup_steps ** -1.5)
        
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        
        # Cập nhật learning rate cho tất cả các param_groups trong optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

    def state_dict(self):
        """Lưu trạng thái của scheduler."""
        return {
            'step_num': self.step_num,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }

    def load_state_dict(self, state_dict):
        """Tải trạng thái của scheduler."""
        self.step_num = state_dict['step_num']
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
