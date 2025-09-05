import torch


class NoiseSchedule:
    """Base class of diffusion noise schedules."""
    def add_noise(self, x, t, noise=None):
        # assuming dimension of t matches first dimension of x
        t_shape = [-1] + [1] * (len(x.shape) - 1)
        tt = t.view(t_shape)
        alpha = self.alpha(tt)
        sigma = self.sigma(tt)
        if noise is None:
            noise = torch.randn_like(x)
        noisy_x = alpha * x + sigma * noise
        return noisy_x, noise

    def denoise_step(self, x, pred_noise, t, dt, z=None, non_stochastic=False):
        # assuming dimension of t matches first dimension of x
        dt = abs(dt)
        t_shape = [-1] + [1] * (len(x.shape) - 1)
        tt = t.view(t_shape)
        dtt = dt.view(t_shape)
        alpha = self.alpha(tt)
        sigma = self.sigma(tt)
        g = self.g(tt)
        g2 = self.g2(tt)
        f = self.f(tt)

        pred_score = -1. * pred_noise / sigma

        if non_stochastic:
            dx = (f * x - 0.5 * g2 * pred_score) * dtt
        else:
            if z is None:
                z = torch.randn_like(x)
            dx = (f * x - g2 * pred_score) * dtt + \
                g * z * torch.sqrt(dtt)
        prev_x = x - dx
        clean_x = (x - sigma * pred_noise) / alpha
        return prev_x, clean_x


class VarianceExplodingNoiseSchedule(NoiseSchedule):
    """
    Variance exploding stochastic differential equation (SDE) scheduler.
    Paper: Song, Yang, et al. "Score-based generative modeling through stochastic differential equations."
    """
    def __init__(self, sigma_max):
        self.sigma_max = sigma_max
        self.alpha = lambda t: torch.ones_like(t)
        self.f = lambda t: torch.zeros_like(t)
        self.sigma = lambda t: t * sigma_max
        self.g2 = lambda t: 2 * sigma_max**2 * t
        self.g = lambda t: self.g2(t)**0.5
        self.dalpha = lambda t: torch.zeros_like(t)
        self.dsigma = lambda t: sigma_max * torch.ones_like(t)


class VariancePreservingNoiseSchedule(NoiseSchedule):
    """
    Variance preserving stochastic differential equation (SDE) scheduler.
    Paper: Song, Yang, et al. "Score-based generative modeling through stochastic differential equations."
    """
    def __init__(self, sigma_max=1.0, schedule='cosine'):
        self.sigma_max = sigma_max
        if schedule == 'cosine':
            self.alpha = lambda t: torch.cos(torch.pi/2*t)
            self.sigma = lambda t: torch.sin(torch.pi/2*t) * sigma_max
            self.f = lambda t: torch.tan(torch.pi/2*t) * torch.pi * (-0.5)
            self.g2 = lambda t: torch.pi * \
                self.alpha(t) * self.sigma(t) * sigma_max - \
                2 * self.f(t) * self.sigma(t)**2
            self.g = lambda t: self.g2(t)**0.5
        elif schedule == 'linear':
            self.gamma = lambda t: 1 - t
            self.alpha = lambda t: self.gamma(t)**0.5
            self.sigma = lambda t: (1 - self.gamma(t))**0.5 * sigma_max
            self.f = lambda t: 0.5 / (t - 1)
            self.g2 = lambda t: (1 - 2*self.f(t)*t) * sigma_max**2
            self.g = lambda t: self.g2(t)**0.5
        else:
            raise NotImplementedError(f'Unknown noise schedule: {schedule}')
