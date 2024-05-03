import torch
import math

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super(CustomSGD, self).__init__(params, lr=lr)

    def step(self, grads=None):
        if grads is None:
            raise ValueError("CustomSGD requires gradients to be provided in the input")
        else:
            for group in self.param_groups:
                for p, g in zip(group['params'], grads):
                    if g is None:
                        continue
                    p.data.add_(-group['lr'], g)

class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, grads):
        for group in self.param_groups:
            for p, g in zip(group['params'], grads):
                if g is None:
                    continue
                grad = g.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                #amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # if amsgrad:
                    #     # Maintains max of all exp. moving avg. of sq. grad. values
                    #     state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # if amsgrad:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                # else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Update the parameters
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
class CustomAdamShort(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, grads):
        for group, g in zip(self.param_groups, grads):
            for p in group['params']:
                grad = g.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)