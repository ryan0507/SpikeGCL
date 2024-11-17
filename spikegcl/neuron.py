from math import pi

import torch
import torch.nn as nn


def reset_net(net: nn.Module, reset_type: str = "subtract"):
    for m in net.modules():
        if hasattr(m, "reset"):
            m.reset(reset_type=reset_type)


def heaviside(x: torch.Tensor):
    return x.ge(0)


def gaussian(x, mu, sigma):
    """
    Gaussian PDF with broadcasting.
    """
    return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (
        sigma * torch.sqrt(2 * torch.tensor(pi))
    )


class BaseSpike(torch.autograd.Function):
    """
    Baseline spiking function.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class SuperSpike(BaseSpike):
    """
    Spike function with SuperSpike surrogate gradient from
    "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.

    Design choices:
    - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
    - alpha scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x.abs()) ** 2
        return grad_input * sg, None


class MultiGaussSpike(BaseSpike):
    """
    Spike function with multi-Gaussian surrogate gradient from
    "Accurate and efficient time-domain classification...", Yin et al. 2021.

    Design choices:
    - Hyperparameters determined through grid search (Yin et al. 2021)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
        sg = (
            1.15 * gaussian(x, zero, alpha)
            - 0.15 * gaussian(x, alpha, 6 * alpha)
            - 0.15 * gaussian(x, -alpha, 6 * alpha)
        )
        return grad_input * sg, None


class TriangleSpike(BaseSpike):
    """
    Spike function with triangular surrogate gradient
    as in Bellec et al. 2020.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(1 - alpha * x.abs())
        return grad_input * sg, None


class ArctanSpike(BaseSpike):
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x * x)
        return grad_input * sg, None


class SigmoidSpike(BaseSpike):
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sgax = (x * alpha).sigmoid_()
        sg = (1.0 - sgax) * sgax * alpha
        return grad_input * sg, None


def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return SuperSpike.apply(x - thresh, alpha)


def mgspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(0.5)):
    return MultiGaussSpike.apply(x - thresh, alpha)


def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return SigmoidSpike.apply(x - thresh, alpha)


def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return TriangleSpike.apply(x - thresh, alpha)


def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return ArctanSpike.apply(x - thresh, alpha)


SURROGATE = {
    "sigmoid": sigmoidspike,
    "triangle": trianglespike,
    "arctan": arctanspike,
    "mg": mgspike,
    "super": superspike,
}


class IF(nn.Module):
    def __init__(
        self, v_threshold=1.0, v_reset=0.0, alpha=1.0, surrogate="triangle", detach=True,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach = detach
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.v = 0.0
        self.reset()

    def reset(self, reset_type: str = "subtract"):
        assert reset_type in ["zero", "subtract"]
        if reset_type == "zero":
            self.v = 0
        else:
            self.v = self.v - self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v += dv
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        if self.detach:
            detached_spike = spike.detach()
            v = self.v.detach()
        else:
            v = self.v
            detached_spike = spike
        # 3. reset
        self.v = (1 - detached_spike) * v + detached_spike * self.v_reset
        return spike

class LIF(nn.Module):
    def __init__(
        self,
        tau=1.0,
        v_threshold=1.0,
        v_reset=0.0,
        alpha=1.0,
        surrogate="triangle",
        detach=True,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach = detach
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.v = 0.0
        self.reset()

    def reset(self, reset_type: str = "subtract"):
        assert reset_type in ["zero", "subtract"]
        if reset_type == "zero":
            self.v = 0
        else:
            self.v = self.v - self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        if self.detach:
            detached_spike = spike.detach()
            v = self.v.detach()
        else:
            v = self.v
            detached_spike = spike
        # 3. reset
        self.v = (1 - detached_spike) * v + detached_spike * self.v_reset
        return spike
        
class PLIF(nn.Module):
    def __init__(
        self,
        tau=1.0,
        v_threshold=1.0,
        v_reset=0.0,
        alpha=1.0,
        surrogate="triangle",
        detach=True,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach = detach
        self.surrogate = SURROGATE.get(surrogate)
        self.register_parameter(
            "tau", nn.Parameter(torch.as_tensor(tau, dtype=torch.float32))
        )
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.v = 0.0

    def reset(self, reset_type: str = "subtract"):
        assert reset_type in ["zero", "subtract"]
        if reset_type == "zero":
            self.v = 0
        else:
            self.v = self.v - self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        if self.detach:
            detached_spike = spike.detach()
            v = self.v.detach()
        else:
            v = self.v
            detached_spike = spike
        # 3. reset
        self.v = (1 - detached_spike) * v + detached_spike * self.v_reset
        return spike

class ALIF(nn.Module):
    def __init__(self,size,  tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle', detach=True):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        self.reset()
        self.v_threshold_values = []  # List to store v_threshold values during forward passes
        

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_th, self.alpha)
        spike = spike.detach()
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
        self.v_th = 0.80 * spike + self.v_th * 0.80
        self.v_th = torch.mean(self.v_th, axis = 0)
        # print(self.v_th.size())
        with torch.no_grad():
            self.v_th = 0.20 * spike.detach() + self.v_th * 0.80
            self.v_th = torch.mean(self.v_th, dim=0)
        
        return spike

    def are_all_v_th_same(self):
        if torch.is_tensor(self.v_th) and len(self.v_threshold_values) > 1:
            # Check if all values in v_threshold_values are close to the first value
            return torch.allclose(self.v_th, torch.Tensor(self.v_threshold_values[0]))
        else:
            # If there's only one value or it's not a tensor, return True
            return True


class BaseNeuron_degree_feat(nn.Module):
    def __init__(self,ssize=128, tau: float =1.0, v_threshold: float=0.25, v_reset: float=0., alpha: float=1.0, 
                 surrogate: str = 'triangle', threshold_trainable : bool = False, init_multi = False):
        
        '''
        tau (float): dacay values for v_tthreshold
        v_thresehold (float): Threshold whether omit spikes or not
        v_reset (float): reet values could be adjusted
        alpha (float): Smoothing Factor for surrogate function
        surrogate (float): Surrogate Functions [simoid, triangle, arctan, super]
        '''
        
        super().__init__()
        self.v_reset = v_reset
        self.v = 0.
        self.train_spike_counts = None
        self.valid_spike_counts = None
        self.train_cur_degree = None
        self.valid_cur_degree = None
        
        # if threshold_trainable:
        # linear_spaced_tensor = torch.linspace(0.0, 30.0, steps=ssize, dtype=torch.float32)
        #     self.register_parameter("v_threshold", nn.Parameter(
        #         torch.as_tensor(linear_spaced_tensor, dtype=torch.float32)
        #     ))
            
        try:
            self.surrogate = SURROGATE[surrogate]
        except:
            print('Unvailable surrogate function. Please check surrogate functions')
         
        
        # Tau and alpha (smoothing factor) should not be updated
        
        # Check v_threshold values for trainable (default: False)
        if isinstance(self, LAPLIF_deg_feat):
            self.register_parameter("tau", nn.Parameter(
                torch.as_tensor(tau, dtype=torch.float32)))
            print('Tau paramter')
        else:
            self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
            print('Tau buffer')
        
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        
        self.v_reset_channel = 0.
        # Reset to Initial input values
        # self.reset() 
        

    def reset(self):
        '''
        Reset all of the Neuron states
        self.v : Tensor[size] Self neuron state that all of the neuron 
        implicitly own itself.
        self.v_th : Set threshold to omit spikes, it cloud to adjust for threshold values
        '''
        self.v = 0.
        
        # if not isinstance(self, LIFboth):
        # self.v_th = self.v_threshold
        
    def calibrated_neuron(self):
        eps = 1e-7
        if type(self.v) is float:
            max_v = 0
            min_v = 0
        else:
            max_v = torch.max(self.v)
            min_v = torch.min(self.v)
            # print(max_v, min_v)
            # print(self.v[self.v > 0].all())
            
        self.v = (self.v - min_v) / (max_v - min_v + eps)
           
    def forward(self, dv):
        raise NotImplementedError
    
    def save_neuron_spikes(self, path):
        torch.save(self.train_spike_counts, f"{path}_train_rate.pt")
        torch.save(self.train_cur_degree, f"{path}_train_cur_degree.pt")
        torch.save(self.valid_spike_counts, f"{path}_valid_rate.pt")
        torch.save(self.valid_cur_degree, f"{path}_valid_cur_degree.pt")
        self.save_threshold(path)
    def update_spike_counts(self, degree, cur_spike):
        
        if self.training:
            self.train_cur_degree = degree
            if self.train_spike_counts is None:
                self.train_spike_counts = cur_spike
            else:
                self.train_spike_counts += cur_spike
        else:
            self.valid_cur_degree = degree
            if self.valid_spike_counts is None:
                self.valid_spike_counts = cur_spike
            else:
                self.valid_spike_counts += cur_spike
        
        return
            
    
    def reset_stat(self):
        self.train_spike_counts = None
        self.valid_spike_counts = None

    def save_threshold(self, path):
        if hasattr(self, 'v_threshold'):
            torch.save(self.v_threshold, f"{path}_v_threshold.pt")
        if hasattr(self, 'v_th'):
            torch.save(self.v_th, f"{path}_v_th.pt")

class LAPLIF_deg_feat(BaseNeuron_degree_feat):
    '''
    Leaky Integrated Fire models with learnable threshold (LAP_LIF type)
    Make the threshold trainable with column-wise
    '''
    
    def __init__(self, ssize=128, tau=1.0, v_threshold=0.25, v_reset=0.0, alpha=1.0, 
                 surrogate='sigmoid', threshold_trainable : bool = False, bins=20, args = None):
        super().__init__(ssize, tau, v_threshold, v_reset, alpha, surrogate, threshold_trainable, bins)
        
        # List to store v_threshold values during forward passes
        self.init_threshold = v_threshold
        self.gamma = 0.20
        self.args = args
        initial_tensor = torch.as_tensor([v_threshold] * bins, dtype=torch.float32)
        v_threshold_tensor = initial_tensor.unsqueeze(1).expand(bins, ssize)
        
        self.register_parameter("v_th", nn.Parameter(
            v_threshold_tensor.clone()
        ))
        
        self.register_parameter("v_threshold", nn.Parameter(
            v_threshold_tensor.clone()
        ))
        # self.register_parameter("v_threshold", nn.Parameter(
                # torch.linspace(0, 2*v_threshold, steps=(bins))
            # ))
        self.v_threshold_values = []
        self.time_step = 0  
        self.reset()
        
    def forward(self, dv, binned_degree, orig_degree=None):
        '''
        dv (Tensor) : input size and output size automatically given
        '''
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # Surrogated -> v_th should not be changed for this neuron
        spike = torch.zeros_like(self.v)
        total_degree = torch.unique(binned_degree).tolist()
        
        if self.time_step == 0:
            for cur_degree in total_degree:
                spike[binned_degree == cur_degree] = self.surrogate(self.v[binned_degree == cur_degree], self.v_threshold[cur_degree], self.alpha)
            self.v_th = nn.Parameter(self.v_threshold.clone().detach())     

            # if self.v_threshold.grad is not None:
            #     print(f"Time step {self.time_step}: v_threshold gradient = {self.v_threshold.grad}")
            # else:
            #     print(f"Time step {self.time_step}: v_threshold gradient = None (no backpropagation yet)")

        else:
            for cur_degree in total_degree:
                spike[binned_degree == cur_degree] = self.surrogate(self.v[binned_degree == cur_degree], self.v_th[cur_degree], self.alpha)
            
        self.v = (1 - spike.detach()) * self.v + spike.detach() * self.v_reset
        # print(self.time_step)
        # print("v_threshold values (Learnable)")
        # print(self.v_threshold)
        # print("v_th values (Adaptive)")
        # print(self.v_th)
        with torch.no_grad():
            v_th_new = self.v_th.clone()
            for i in range(self.v_th.size(0)):
                mask = (binned_degree == i)
                if mask.any():
                    v_th_new[i] = self.gamma * spike[mask].mean(axis=0) + self.v_th[i] * (1 - self.gamma)
        
        # print(f"Time step {self.time_step}: v_th (after update) = {self.v_th.detach().cpu().numpy()}")
        self.v_th = nn.Parameter(v_th_new)
        self.time_step += 1
        # print(self.v_th)
        
        # if self.v_th.grad is not None:
        #     print(f"Time step {self.time_step}: v_th gradient = {self.v_th.grad}")
        # else:
        #     print(f"Time step {self.time_step}: v_th gradient = None (no backpropagation due to torch.no_grad())")
        if self.training:
            if torch.is_tensor(self.v_th):
                mean_val = torch.mean(self.v_th)            
                self.v_threshold_values.append(mean_val.item())
        
        updated_banned_datasets = ['ogbg-molhiv', 'ogbg-ppa', 'reddit-binary', 'collab', 'zinc', 'zinc-500k']
        # if 'ugformer' not in self.args.model.lower() and self.args.dataset.lower() not in updated_banned_datasets:    
            # self.update_spike_counts(orig_degree, spike)
        
        return spike
    
    def reset(self):
        '''
        Reset all of the Neuron states
        self.v : Tensor[size] Self neuron state that all of the neuron 
        implicitly own itself.
        self.v_th : Set threshold to omit spikes, it cloud to adjust for threshold values
        '''
        self.v = 0.
        self.time_step = 0  
        self.v_th = nn.Parameter(self.v_threshold.clone().detach()) 