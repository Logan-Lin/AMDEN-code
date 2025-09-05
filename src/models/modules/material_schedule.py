import numpy as np
import torch
from tqdm import tqdm

from .noise_schedules import VarianceExplodingNoiseSchedule, VariancePreservingNoiseSchedule


class MaterialNoiseSchedule:
    """Compound noise schedule for diffusing both atomic positions and elements.
    
    Manages the forward and reverse diffusion processes for material structures,
    handling both continuous (positions) and discrete (elements) features. Uses
    variance exploding (VE) schedule for positions to preserve scale invariance
    and optionally variance preserving (VP) schedule for element embeddings.
    
    This scheduler is the core of the material diffusion model, orchestrating
    the gradual addition of noise (forward process) and iterative denoising
    (reverse process) to generate new material structures.
    
    Attributes:
        model (nn.Module): The denoiser model for predicting noise.
        uncond_model (nn.Module, optional): Unconditional model for classifier-free guidance.
        t_min (float): Minimum time value (typically near 0).
        t_max (float): Maximum time value (typically 1).
        pos_noise (VarianceExplodingNoiseSchedule): Schedule for position diffusion.
        el_noise (VariancePreservingNoiseSchedule, optional): Schedule for element diffusion.
    """
    
    def __init__(self, model, t_min, t_max, uncond_model=None, sigma_max_pos=1.0,
                 noise_schedule_el=None, sigma_max_el=1.0):
        """Initialize material noise scheduler.

        Args:
            model (nn.Module): Denoiser model for noise prediction.
            t_min (float): Start time of noise schedule (typically 0.001).
            t_max (float): End time of noise schedule (typically 1.0).
            uncond_model (nn.Module, optional): Model for unconditional generation.
            sigma_max_pos (float, optional): Maximum noise std for positions. Defaults to 1.0.
            noise_schedule_el (str, optional): Schedule type for elements ('linear', 'cosine', or None).
                If None, elements are not diffused. Defaults to None.
            sigma_max_el (float, optional): Maximum noise std for elements. Defaults to 1.0.
        """
        # super().__init__()

        self.model = model
        self.uncond_model = uncond_model
        self.t_min = t_min
        self.t_max = t_max

        self.pos_noise = VarianceExplodingNoiseSchedule(sigma_max_pos)
        self.el_noise = None

        if noise_schedule_el is not None:
            if model.element_embedding is not None:
                self.el_noise = VariancePreservingNoiseSchedule(
                    sigma_max_el, noise_schedule_el)
            else:
                raise Exception(
                    "Diffusion of elements is only possible in combination with an element embedding")

    def add_noise(self, clean_sample, t):
        """Add noise to a batch of clean samples based on the given timestamps.

        Args:
            - clean_sample (Sample): a batch of clean material samples.
            - t (LongTensor): a batch of timestamps, with shape (batch_size).

        Returns:
            - Sample: the batch of noisy samples.
        """
        t_per_at = t[clean_sample.get_batch_indices()]
        noise_pos = torch.randn_like(clean_sample.get_positions())          
        noise_pos = clean_sample.remove_mean(noise_pos)
        noisy_pos, noise_pos = self.pos_noise.add_noise(
            clean_sample.get_positions(), t_per_at, noise=noise_pos)

        if self.model.element_embedding is not None:
            clean_el_emb = clean_sample.get_element_emb()
            if clean_el_emb is None:
                clean_el_emb = self.model.element_embedding.embed(
                    clean_sample.get_elements())
            if self.el_noise is not None:
                noisy_els, noise_els = self.el_noise.add_noise(clean_el_emb, t[clean_sample.get_batch_indices()])
            else:
                noise_els = None
                noisy_els = clean_el_emb  # we don't noise the elements
        else:
            noise_els = None
            noisy_els = None

        noisy_sample = clean_sample.update_attrs(
            positions=noisy_pos,
            element_emb=noisy_els,
            elements=None if self.el_noise is None else self.model.element_embedding.unembed(
                noisy_els)
        )


        return noisy_sample, (noise_pos, noise_els)

    def calc_dlnp_dt(self, clean_sample, noise_pos, noise_els, t):
        t_per_at = t[clean_sample.get_batch_indices().to(t.device)]
        # According to my calculations: 
        # d/dt ln(p(x_t, t)) = 1 / sigma_t (x_0 alpha' eps + sigma' eps^2 - sigma')
        # note how (if alpha'=0) we get sigma'(eps^2-1) ... on average eps^2=1 by definition -> this keeps average value of lnlp constant!
        dsigma = self.pos_noise.dsigma(t_per_at.unsqueeze(-1))
        dlnp_dt_pos = torch.sum((noise_pos**2 - 1) * dsigma, dim=-1)
        dlnp_dt_pos = dlnp_dt_pos / self.pos_noise.sigma(t_per_at)
        dlnp_dt_pos = torch.zeros_like(t).index_add(0, clean_sample.get_batch_indices().to(t.device), dlnp_dt_pos)
        dlnp_dt = dlnp_dt_pos 
        return dlnp_dt


    def denoise_step(self, noisy_sample, t, dt, non_stochastic=False, con_weight=0.0):
        """Predict the sample from the previous timestep by reversing the SDE.

        Args:
            - noisy_sample (Sample): the batch of samples in the current diffusion timesteps.
            - t (float): the current diffusion timesteps.
            - dt (float): diffusion step size.
            - non_stochastic (bool, optional): If False, the stochastic denoising method is used. Defaults to False.

        Returns:
            - Sample: computed sample (x_{t-1}) of previous timestep.
            - Sample: predicted denoised sample (x_{0}) based on the model output from the current timestep.
        """

        t_tensor = torch.tensor((t,)).float().to(noisy_sample.get_positions().device)
        t_tensor_batch = t_tensor.repeat(noisy_sample.get_batch_size())
        dt_tensor = torch.tensor((dt,)).float().to(noisy_sample.get_positions().device)
        dt_tensor_batch = dt_tensor.repeat(noisy_sample.get_batch_size())

        con_noise_pos, con_noise_els = self.model(noisy_sample, t_tensor_batch)
        if con_weight is None or con_weight == 0.0:
            # Vanilla reverse diffusion process.
            pred_noise_pos, pred_noise_els = con_noise_pos, con_noise_els
        else:
            # The reverse diffusion process under regressor-free guidance.
            if self.uncond_model is None:
                uncon_noise_pos, uncon_noise_els = self.model(noisy_sample.null_properties(), t_tensor_batch)
            else:
                uncon_noise_pos, uncon_noise_els = self.uncond_model(noisy_sample.null_properties(), t_tensor_batch)
            pred_noise_pos = (1.0 + con_weight) * con_noise_pos - con_weight * uncon_noise_pos
            pred_noise_els = (1.0 + con_weight) * con_noise_els - con_weight * uncon_noise_els

        # Denoising of positions.
        z = torch.randn_like(noisy_sample.get_positions())
        z = noisy_sample.remove_mean(z)
        new_positions, clean_positions = self.pos_noise.denoise_step(noisy_sample.get_positions(
        ), pred_noise_pos, t_tensor, dt_tensor, z, non_stochastic=non_stochastic)

        # Denoising of elements.
        new_element_emb = None
        if self.el_noise is not None:
            new_element_emb, clean_element_emb = self.el_noise.denoise_step(
                noisy_sample.get_element_emb(), pred_noise_els, t_tensor, dt_tensor, non_stochastic=non_stochastic)

        pred_prev_sample = noisy_sample.update_attrs(
            positions=new_positions,
            elements=self.model.element_embedding.unembed(
                new_element_emb) if self.el_noise is not None else None,
            element_emb=new_element_emb if self.el_noise is not None else None
        )
        pred_clean_sample = noisy_sample.update_attrs(
            positions=clean_positions,
            elements=self.model.element_embedding.unembed(
                clean_element_emb) if self.el_noise is not None else None,
            element_emb=clean_element_emb if self.el_noise is not None else None
        )

        return pred_prev_sample, pred_clean_sample

    def denoise(self, noisy_sample, n_steps,
                t_min=None, t_max=None, non_stochastic=False, con_weight=0.0, final_step=True,
                hmc_n_iterations=0, hmc_n_steps=15, hmc_dt=0.5, hmc_range=(0., 1.),
                log_file=None, tqdm_desc='Denoising'):
        """Perform the complete denoising of a given sample/batch

        Args:
            - noisy_sample (Sample/Batch): Sample to denoise
            - n_steps (int): Number of denoising steps
            - t_min (int, optional): Final time of the denoising. If none, the schedulers default is used.
            - t_max (int, optional): First time of the denoising. If none, the schedulers default is used.
            - non_stochastic (bool, optional): If False, the stochastic denoising method is used. Defaults to False.
            - con_weight (float, optional): Weight factor for regressor free guidance. Defaults to 0.0.
            - final_step (bool, optional): Wether a final denoising step going from t_min to 0 is added at the end of the trajectory. Defaults to True.

        Returns:
            - list(Sample/Batch): The complete denoising trajectory
        """
        t_min = self.t_min if t_min is None else t_min
        t_max = self.t_max if t_max is None else t_max
        timesteps = self.gen_infer_timesteps(n_steps, t_min, t_max)
        dt = abs(timesteps[0] - timesteps[1])

        sample = noisy_sample
        trajectory = [sample]

        log_lines = []
        for i, t in tqdm(enumerate(timesteps), leave=False, total=n_steps, desc=tqdm_desc):
            # do HMC to equilibrate at time t
            if t > hmc_range[0] and t < hmc_range[1]:
                for i_hmc in range(hmc_n_iterations):
                    # the timestep is scaled with sigma_t
                    sample, hmc_acc, energy = self.hmc_iteration(sample, t, hmc_n_steps, hmc_dt * self.pos_noise.sigma(t))

                    if log_file is not None:
                        with open(log_file, 'a') as f:
                            for batch_i, (acc, e) in enumerate(zip(hmc_acc.detach().cpu().numpy(), energy.detach().cpu().numpy())):
                                log_lines.append(f"HMC-ACC {i} {t} {batch_i} {i_hmc} {acc} {e}\n")

            # decrease t
            pred_prev_sample, _ = self.denoise_step(
                noisy_sample=sample,
                t=t,
                dt=dt,
                non_stochastic=non_stochastic,
                con_weight=con_weight)
            sample = pred_prev_sample

            trajectory.append(sample)

            # add energy to log file if we're doing HMC
            if log_file is not None:
                if hmc_n_iterations > 0:
                    t_tensor = torch.tensor((t - dt,)).float().to(sample.get_positions().device).repeat(sample.get_batch_size())
                    sigma_t = self.pos_noise.sigma(torch.tensor((t-dt,)).float().to(sample.get_positions().device).repeat(sample.get_batch_size()))
                    energy, _, _ = self.model.get_energy_and_forces(sample, t_tensor, sigma_t, calc_energy=True, calc_forces=False, calc_dE_del=False)
                    for batch_i, e in enumerate(energy.detach().cpu().numpy()):
                        log_lines.append(f"ENERGY {i} {t-dt} {batch_i} {e}\n")

        if log_file is not None:
            with open(log_file, 'a') as f:
                f.writelines(log_lines)

        if final_step:
            pred_prev_sample, _ = self.denoise_step(
                noisy_sample=sample,
                t=t_min,
                dt=t_min,
                non_stochastic=non_stochastic,
                con_weight=con_weight)
            sample = pred_prev_sample
            trajectory.append(sample)

        return trajectory

    def hmc_iteration(self, sample, t, n_steps, dt):
        # preparing helper stuff
        batch_idx = sample.get_batch_indices().to(sample.get_positions().device)
        batch_size = sample.get_batch_size()
        t_tensor = torch.tensor((t,)).float().to(sample.get_positions().device)
        sigma_t = self.pos_noise.sigma(torch.tensor((t,)).float().to(sample.get_positions().device).repeat(batch_size))

        x = sample
        # Initialize momenta from Boltzman distribution (assuming masses and temperature are 1)
        p = torch.randn_like(x.get_positions()) 
        e_pot_init, f, _ = self.model.get_energy_and_forces(x, t_tensor, sigma_t, calc_energy=True, calc_forces=True, calc_dE_del=False)
        e_tot_init = e_pot_init.index_add(0, batch_idx, 0.5 * torch.sum(p**2, dim=1))
        # velocit Verlet <3
        for i in range(n_steps): 
            x = x.update_attrs(positions=x.get_positions() + p * dt + 0.5 * f * dt**2)
            f_last = f
            _, f, _ = self.model.get_energy_and_forces(x, t_tensor, sigma_t, calc_energy=False, calc_forces=True, calc_dE_del=False)
            p = p + (f + f_last) / 2 * dt

        e_pot_final, _, _ = self.model.get_energy_and_forces(x, t_tensor, sigma_t, calc_energy=True, calc_forces=False, calc_dE_del=False)
        e_tot_final = e_pot_final.index_add(0, batch_idx, 0.5 * torch.sum(p**2, dim=1))
        p_accept = torch.exp(-(e_tot_final - e_tot_init))
        accept = torch.rand(batch_size).to(p_accept.device) < p_accept

        pos = sample.get_positions() # initial positions
        pos[accept[batch_idx]] = x.get_positions()[accept[batch_idx]] # update positions of accepted structures
        e_pot_final[~accept] = e_pot_init[~accept] 
        x = x.update_attrs(positions=pos)
        return x, accept, e_pot_final


    def gen_random_t(self, batch):
        t = torch.rand((batch.get_batch_size(),)).float().to(
            batch.get_positions().device)
        t = t * (self.t_max - self.t_min) + self.t_min
        return t

    def gen_random_sample(self, sample):
        emb = None
        elements = None
        if self.model.element_embedding is not None:
            emb = self.model.element_embedding.embed(sample.get_elements())
            if self.el_noise is not None:
                emb = torch.randn_like(emb) * self.el_noise.sigma_max
            elements = self.model.element_embedding.unembed(emb)
        return sample.randomize_uniform().update_attrs(element_emb=emb, elements=elements)

    # generates the time stamps for the denoising. t_min is excluded.
    def gen_infer_timesteps(self, n_steps, t_min=None, t_max=None):
        t_min = self.t_min if t_min is None else t_min
        t_max = self.t_max if t_max is None else t_max
        return np.linspace(1, 0, n_steps + 1)[:-1] * (t_max - t_min) + t_min
