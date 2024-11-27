import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import tqdm

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)*x.size(2)
    x = x.reshape(x_size,-1).unsqueeze(1) # (x_size, 1, dim)
    y = y.reshape(y_size,-1).unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def mmd_fn(real, fake, gamma=1):
    """ Maximum mean discrepancy (MMD) with
        radial basis function (RBF) kernel.
        Instead of returning MMD, this function return the
        -log(MMD) easy tracking.

    Args:
        real (batch, seq_len, feature_size): Real sequence.
        fake (batch, seq_len, feature_size): Fake sequence.
        gamma (float): RBF constant

    Returns:
        output (float): Computed MMD with RBF.
    """
    Br, Bf = real.shape[0], fake.shape[0]
    real = real.reshape(Br, -1)
    fake = fake.reshape(Bf, -1)

    Krr = (-gamma * torch.cdist(real, real).pow(2)).exp().sum() - Br
    Krf = (-gamma * torch.cdist(real, fake).pow(2)).exp().sum()
    Kff = (-gamma * torch.cdist(fake, fake).pow(2)).exp().sum() - Bf

    output = -((1/(Br*(Br-1)))*Krr - (2/(Br*Bf))*Krf + (1/(Bf*(Bf-1)))*Kff).abs().log()
    return output

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, schedule_name = "linear",loss_type='mse' ):
        super().__init__()

        self.model = model
        self.T = T
        self.loss_type = loss_type
        if schedule_name == "linear":
            self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double()) #产生一个linear schedule, 为betas
        elif schedule_name == "cosine":
            self.register_buffer('betas', cosine_beta_schedule(T, s = 0.008))
        alphas = 1. - self.betas #得到论文中的α
        alphas_bar = torch.cumprod(alphas, dim=0) #得到论文中的α_bar

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)) 
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
    def forward(self, x_0, labels):
        """
        Algorithm 1., x_0 一般也被称作x_start
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        pred_noise = self.model(x_t, t, labels)
        if self.loss_type == 'mse+mmd':
            loss = F.mse_loss(pred_noise, noise, reduction='none') + 0.1 * mmd_fn(noise, pred_noise)
        else:
            loss = F.mse_loss(pred_noise, noise, reduction='none') 
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0., schedule_name = "linear" ):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        if schedule_name == "linear":
            self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        elif schedule_name == "cosine":
            self.register_buffer('betas', cosine_beta_schedule(T, s = 0.008))
        alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:T] #alphas_t-1
        self.register_buffer('coeff1', torch.sqrt(1. / alphas)) #得到论文中的1/sqrt(α_bar)
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - self.alphas_bar)) 
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - self.alphas_bar)) #方差的上限

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps #noise
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   

    def sample_backward(self,
                        img_or_shape,
                        device,
                        label,
                        simple_var=True,
                        ddim_step=30,
                        eta=1):
        if simple_var:
            eta = 1
        ts = torch.linspace(self.T, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
        batch_size = x.shape[0]
        for i in range(1, ddim_step + 1):
            print(i)
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alphas_bar[cur_t]
            ab_prev = self.alphas_bar[prev_t] if prev_t >= 0 else 1

            t_tensor = torch.tensor([cur_t] * batch_size,
                                    dtype=torch.long).to(device)
            eps = self.model(x, t_tensor, label)

            if self.w != 0:
                nonEps = self.model(x, t_tensor, torch.zeros_like(label).to(label.device))
                eps = (1. + self.w) * eps - self.w * nonEps #noise

            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term

        return x

    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img