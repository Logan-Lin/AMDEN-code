import torch


class NoiseSchedule:
    """Base class for noise schedules in diffusion models.
    
    Defines the interface for forward and reverse diffusion processes,
    including methods for adding noise, computing noise levels, and
    their derivatives. Subclasses implement specific noise schedules
    like variance exploding (VE) or variance preserving (VP).
    
    The noise schedule determines how noise is added during the forward
    process and how it's removed during the reverse (denoising) process.
    """
    def add_noise(self, x, t, noise=None):
        """Add noise to clean data according to the schedule at time t.
        
        Args:
            x (torch.Tensor): Clean data tensor.
            t (torch.Tensor): Time steps, shape matching x's batch dimension.
            noise (torch.Tensor, optional): Pre-generated noise. If None, generates new noise.
            
        Returns:
            tuple: (noisy_x, noise) where noisy_x is the noised data and noise is the added noise.
        """
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
        """Perform one step of the reverse diffusion process.
        
        Args:
            x (torch.Tensor): Current noisy data.
            pred_noise (torch.Tensor): Predicted noise from the denoiser model.
            t (torch.Tensor): Current time step.
            dt (float): Time step size (negative for reverse process).
            z (torch.Tensor, optional): Random noise for stochastic sampling.
            non_stochastic (bool, optional): If True, use deterministic ODE instead of SDE.
            
        Returns:
            tuple: (prev_x, clean_x) where prev_x is the denoised sample at t-dt
                and clean_x is the estimated clean sample.
        """
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
    """Variance Exploding (VE) noise schedule for diffusion models.
    
    In VE schedules, the signal remains unchanged (alpha=1) while noise
    variance increases linearly with time. This is particularly suitable
    for position coordinates in material systems where absolute scale matters.
    
    The noise level grows from 0 at t=0 to sigma_max at t=1.
    
    Reference: Song et al., "Score-based generative modeling through 
               stochastic differential equations" (NeurIPS 2021)
    """
    def __init__(self, sigma_max):
        """Initialize VE noise schedule.
        
        Args:
            sigma_max (float): Maximum noise standard deviation at t=1.
        """
        self.sigma_max = sigma_max
        self.alpha = lambda t: torch.ones_like(t)
        self.f = lambda t: torch.zeros_like(t)
        self.sigma = lambda t: t * sigma_max
        self.g2 = lambda t: 2 * sigma_max**2 * t
        self.g = lambda t: self.g2(t)**0.5
        self.dalpha = lambda t: torch.zeros_like(t)
        self.dsigma = lambda t: sigma_max * torch.ones_like(t)


class VariancePreservingNoiseSchedule(NoiseSchedule):
    """Variance Preserving (VP) noise schedule for diffusion models.
    
    In VP schedules, the total variance (signal + noise) is preserved
    throughout the diffusion process. The signal is gradually attenuated
    while noise is added to maintain unit variance. This is suitable for
    normalized features like element embeddings.
    
    Supports 'cosine' and 'linear' interpolation schedules between
    clean data (t=0) and pure noise (t=1).
    
    Reference: Song et al., "Score-based generative modeling through
               stochastic differential equations" (NeurIPS 2021)
    """
    def __init__(self, sigma_max=1.0, schedule='cosine'):
        """Initialize VP noise schedule.
        
        Args:
            sigma_max (float, optional): Maximum noise scale. Defaults to 1.0.
            schedule (str, optional): Interpolation schedule ('cosine' or 'linear').
                'cosine' provides smoother transitions, 'linear' is more uniform.
                Defaults to 'cosine'.
        """
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
