import os

import numpy as np
import torch
from ase.io import write as write_ase
from tqdm import trange, tqdm
import json

import utils
from data import save_dirs


def train(model, scheduler, loss_func, train_dataloader,
          lr, num_epochs, start_epoch=0, prop_null_prob=0.0, prop_null_prob_all=0.0, 
          optimizer_name='adam', lr_scheduler=None,
          save=None, orig_model=None, init_r_cut=None, clip_grad_norm=None):
    """Training pipeline.

    Args:
        scheduler: diffusion scheduler.
        loss_func: diffusion loss function.
        model (nn.Module): the diffusion denoiser model to train.
        train_dataloader (DataLoader): data iterator for training.
        lr (float): the initial learning rate of the optimizer.
        num_epochs (int): total number of epoches to train the model.
        prop_null_prob (float, optional): probability of a property randomly masked into a null property.
        optimizer_name (str, optional): indicating which optimizer to use. Defaults to 'adam'.
        lr_scheduler (dict, optional): dictionary containing scheduler configuration with keys:
            - 'enabled': bool, whether to use the scheduler
            - 'name': str, name of the scheduler ('reduce_lr_on_plateu' or 'one_cycle_lr')
            - 'params': dict, parameters to pass to the scheduler
        save (dict, optional): dictionary containing save configuration with keys:
            - 'name': str, name for saving log and model parameters
            - 'epoch': int, frequency of saves
        orig_model (nn.Module, optional): the uncompiled model.
        init_r_cut: initial cutoff distance.
        clip_grad_norm (float, optional): clip the gradient norm of model parameters.
    """
    if init_r_cut is not None:
        for sample in tqdm(train_dataloader.dataset, desc='Creating initial neighborlists'):
            sample.update_edges(init_r_cut)

    # Choose the optimizer to use.
    if optimizer_name == 'adam':
        OptimizerClass = torch.optim.Adam
    elif optimizer_name == 'sgd':
        OptimizerClass = torch.optim.SGD
    else:
        raise NotImplementedError(f'No optimizer called {optimizer_name}')
    optimizer = OptimizerClass(list(model.parameters()), lr=lr)

    # Choose the learning rate scheduler to use.
    if lr_scheduler is not None and lr_scheduler.get('enabled', False):
        if lr_scheduler['name'] == 'reduce_lr_on_plateu':
            lr_scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **lr_scheduler['params'])
        elif lr_scheduler['name'] == 'one_cycle_lr':
            lr_scheduler_obj = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, total_steps=num_epochs * len(train_dataloader), **lr_scheduler['params'])
        else:
            raise NotImplementedError(f'No lr scheduler {lr_scheduler["name"]}')

    model.train()

    # Prepare the log file.
    os.makedirs(os.path.dirname(f"{save_dirs['log']}/{save['name']}"), exist_ok=True)
    log_path = utils.next_version(save_dirs['log'], save['name'] + '-train', '.log') if save else None
    train_logger = utils.TrainLogger(log_path, 10, scheduler.t_min,
                                     scheduler.t_max, log_split_loss=scheduler.el_noise is not None)
    tb_logger = utils.TensorBoardLogger(os.path.join(save_dirs['log'], 'tensorboard', save['name'])) if save else None

    # The implementation of the dpdt loss is a bit messy
    # It probably won't work so we'll remove it anyways
    # Just getting the weight from the loss_func and then doing everything here
    use_dpdt = abs(loss_func.dpdt_weight) > 1.e-10

    epoch_loss, num_nan = 0, 0

    desc = 'Avg loss %.6f, num nan %d'

    with trange(start_epoch, num_epochs, desc=desc % (0, 0)) as bar:
        for epoch in bar:
            loss_values = []
            for batch_i, clean_sample in enumerate(tqdm(train_dataloader, leave=False)):
                # The diffusion training process.
                # Randomly generate the diffusion timestep,
                # and obtain the noisy sample at that timestep through the forward process.
                t = scheduler.gen_random_t(clean_sample)

                if use_dpdt:
                    # We train using differences in dpdt (because there is a constant offset)
                    # We therefore need two samples at the same timestep
                    batchsize = clean_sample.get_batch_size()
                    halfbatchsize = batchsize // 2
                    t[:halfbatchsize] = t[halfbatchsize:2*halfbatchsize]

                noisy_sample, (noise_pos, noise_els) = scheduler.add_noise(clean_sample, t)

                # Randomly mask the properties of a portion of samples to null.
                if prop_null_prob > 0.0 or prop_null_prob_all > 0.0:
                    noisy_sample = noisy_sample.null_properties_random(prop_null_prob, prop_null_prob_all)

                # Typically NaN only appears during the forward process of the deriv denoiser.
                try:
                    # Make the denoiser predict noise at this step/noisy sample at previous step/clean sample.
                    pred_noise_pos, pred_noise_els = model(noisy_sample, t)
                except ValueError as e:
                    optimizer.zero_grad()
                    num_nan += 1
                    bar.set_description(desc % (epoch_loss, num_nan))
                    continue

                if use_dpdt:
                    pred_dlnp_dt = model.get_dlnp_dt(noisy_sample, t, scheduler.pos_noise.sigma)
                    dlnp_dt = scheduler.calc_dlnp_dt(clean_sample, noise_pos, noise_els, t)
                    sigma = scheduler.pos_noise.sigma(t)
                    pred_dlnp_dt = pred_dlnp_dt * sigma
                    dlnp_dt = dlnp_dt * sigma

                per_sample_losses = []
                batch_idx = clean_sample.get_batch_indices()
                batch_size = clean_sample.get_batch_size()
                for i_sample in range(batch_size):
                    # Calculate the loss of this step.
                    sample_mask = (batch_idx == i_sample)
                    sample_loss_tot, (sample_loss_pos, sample_loss_el) = loss_func(
                        noise_pos=noise_pos[sample_mask],
                        pred_noise_pos=pred_noise_pos[sample_mask],
                        noise_els=noise_els[sample_mask] if noise_els is not None else None,
                        pred_noise_els=pred_noise_els[sample_mask] if noise_els is not None else None)
                    per_sample_losses.append(sample_loss_tot)

                    train_logger.register_sample_loss(
                        t[i_sample].detach().item(),
                        sample_loss_tot.detach().item(),
                        sample_loss_pos.detach().item() if noise_els is not None else None,
                        sample_loss_el.detach().item() if noise_els is not None else None)

                    if tb_logger is not None:
                        sample_idx = (epoch * len(train_dataloader) + batch_i) * batch_size + i_sample
                        tb_logger.log_scalar('train/sample_loss', sample_loss_tot.item(), sample_idx)
                        if noise_els is not None:
                            tb_logger.log_scalar('train/sample_loss_pos', sample_loss_pos.item(), sample_idx)
                            tb_logger.log_scalar('train/sample_loss_el', sample_loss_el.item(), sample_idx)

                # Backward propagation.
                loss = torch.mean(torch.stack(per_sample_losses))

                if use_dpdt:
                    dpdt_pred = pred_dlnp_dt[:halfbatchsize] - pred_dlnp_dt[halfbatchsize:2*halfbatchsize]
                    dpdt = dlnp_dt[:halfbatchsize] - dlnp_dt[halfbatchsize:2*halfbatchsize]
                    dpdtloss = torch.mean((dpdt - dpdt_pred)**2)
                    loss = loss + loss_func.dpdt_weight * dpdtloss

                optimizer.zero_grad()
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_grad_norm)
                optimizer.step()
                if lr_scheduler is not None and lr_scheduler.get('enabled', False) and lr_scheduler['name'] == 'one_cycle_lr':
                    lr_scheduler_obj.step()
                loss_values.append(loss.detach().item())

                if tb_logger is not None:
                    tb_logger.log_scalar('train/batch_loss', loss.item(), epoch * len(train_dataloader) + batch_i)

            epoch_loss = np.mean(loss_values)
            if lr_scheduler is not None and lr_scheduler.get('enabled', False) and lr_scheduler['name'] == 'reduce_lr_on_plateu':
                lr_scheduler_obj.step(epoch_loss)
            bar.set_description(desc % (epoch_loss, num_nan))
            if tb_logger:
                tb_logger.log_scalar('train/epoch_loss', epoch_loss, epoch)

            lr = optimizer.param_groups[-1]['lr']
            train_logger.log(epoch, lr)
            if tb_logger:
                tb_logger.log_scalar('train/learning_rate', lr, epoch)

            if save is not None and save.get('epoch', None) is not None:
                if (epoch + 1) % save['epoch'] == 0:
                    save_model(model, save['name'], orig_model, epoch + 1)
                    if tb_logger:
                        tb_logger.log_model_params(model, epoch)
    if save is not None:
        save_model(model, save['name'], orig_model)

    # Close TensorBoard logger
    if tb_logger:
        tb_logger.close()


@torch.no_grad()
def infer(model, scheduler, infer_dataloader, n_steps, 
          uncond_model=None, save=None, from_random_noise=True, non_stochastic=False,
          max_repeats=0, max_restarts=0, restart_t=None, restart_n_steps=None,
          con_weight=0.0, hmc_n_iterations=0, hmc_n_steps=15, hmc_dt=0.5, hmc_range=(0., 1.),
          init_r_cut=None):
    """Inference pipeline for generating new material samples.

    Args:
        model (nn.Module): the diffusion denoiser model.
        infer_dataloader (DataLoader): data iterator for inference.
        n_steps (int): number of diffusion steps used for inference.
        save (dict, optional): dictionary containing saving configuration with keys:
            - 'name': str, name for saving generated samples and logs
            - 'trajectories': dict, trajectory saving configuration with keys:
                - 'enabled': bool, whether to save trajectory files
                - 'batches': list, indices of batches to save trajectories for (default: [0])
            - 'enable_log': bool, whether to save log files (default: False)
        from_random_noise (bool, optional): whether to create the starting point of the denoising process by sample from a random distribution 
            or by adding noise to real samples.
    """
    def restart_unbalanced_samples(final_sample, max_attempts, from_random_noise=False):
        """
        Takes the final_sample (Batch) from the original denoising. Checks charge balance.
        For unbalanced samples, re-add noise at time = restart_t, then do a partial denoising.
        If from_random_noise is True, we sample from a random distribution instead of adding noise 
        (basically a repeat of the original denoising).

        Args:
            final_sample (Batch): the final denoised sample from the original run.
            max_attempts (int, optional): maximum number of attempts to restart or repeat the denoising process.
            from_random_noise (bool, optional): whether to create the starting point of the denoising process by sample from a random distribution 
                or by adding noise to real samples.
        Returns:
            final_sample (Batch): updated with any re-denoised unbalanced members.
        """
        partial_n_steps = restart_n_steps if restart_n_steps is not None else n_steps
        device = final_sample.get_positions().device
        num_attempts = [0 for _ in range(final_sample.get_batch_size())]
        restart_trajectories = []
        tqdm_desc = 'repeat' if from_random_noise else 'restart'

        for attempt in range(max_attempts):
            balance_mask = final_sample.is_charge_balanced()
            # If all are balanced, we can stop
            if balance_mask.all():
                break
            
            unbalanced_indices = (~balance_mask).nonzero(as_tuple=True)[0]
            unbalanced_sub_batch = final_sample.get_sub_batch(unbalanced_indices)
            # Re-add noise at time = restart_t only to the unbalanced sub-batch
            t_tensor = torch.ones(unbalanced_sub_batch.get_batch_size(), device=device) * restart_t
            if from_random_noise:
                noisy_unbalanced_sub_batch = scheduler.gen_random_sample(unbalanced_sub_batch)
            else:
                noisy_unbalanced_sub_batch, _ = scheduler.add_noise(unbalanced_sub_batch, t_tensor)

            # Now do a partial denoising from t=restart_t -> 0
            # We can exploit the denoise() function by passing t_min=0, t_max=restart_t, 
            # so it runs from restart_t down to 0. 
            trajectory_partial = scheduler.denoise(noisy_unbalanced_sub_batch, 
                                                  n_steps if from_random_noise else partial_n_steps,
                                                  t_max=None if from_random_noise else restart_t,
                                                  con_weight=con_weight,
                                                  non_stochastic=non_stochastic,
                                                  hmc_n_iterations=hmc_n_iterations,
                                                  hmc_n_steps=hmc_n_steps,
                                                  hmc_dt=hmc_dt,
                                                  hmc_range=hmc_range,
                                                  tqdm_desc=f'{tqdm_desc} {attempt}')

            # The final sub-batch after partial denoising
            re_denoised_unbalanced = trajectory_partial[-1]
            final_sample.update_sub_batch(unbalanced_indices, re_denoised_unbalanced)

            # Save restart trajectories if enabled
            if save_name is not None and enable_trajectories and (save_traj_batches == 'all' or batch_i in save_traj_batches):
                # Convert trajectory to ASE format
                ase_traj = [[at.to_ase_atoms() for at in batch.samples] for batch in trajectory_partial]
                for batch in ase_traj:
                    for at in batch:
                        at.wrap()
                
                # Save trajectory for each unbalanced sample
                for idx, orig_idx in enumerate(unbalanced_indices):
                    traj = [s[idx] for s in ase_traj]
                    # Use the original sample index in the filename
                    sample_idx = len(all_inferred_samples) + orig_idx
                    write_ase(os.path.join(save_dir, f'traj-{sample_idx:05d}_{tqdm_desc}-{attempt}.extxyz'), traj)

            for orig_idx in unbalanced_indices:
                num_attempts[orig_idx] = attempt

        if tb_logger is not None:
            for idx, num_attempt in enumerate(num_attempts):
                tb_logger.log_scalar(f'infer/num_{tqdm_desc}', num_attempt, len(all_inferred_samples) + idx)

        return final_sample

    model.eval()
    if uncond_model is not None:
        uncond_model.eval()

    save_name = save.get('name') if save else None
    traj_config = save.get('trajectories', {}) if save else {}
    enable_trajectories = traj_config.get('enabled', False)
    save_traj_batches = traj_config.get('batches', [0]) if enable_trajectories else []
    enable_log = save.get('enable_log', False) if save else False

    if save_name is not None:
        save_dir = f"{save_dirs['infer']}/{save_name}"
        os.makedirs(save_dir, exist_ok=True)
        tb_logger = utils.TensorBoardLogger(os.path.join(save_dirs['log'], 'tensorboard', save_name)) if enable_log else None
        hmc_logging_path = f"{save_dirs['log']}/hmc_logs/{save_name}.txt"
        os.makedirs(os.path.dirname(hmc_logging_path), exist_ok=True)
        with open(hmc_logging_path, 'w') as f:
            f.write("type time_i time batch_i i_hmc hmc_acc energy\n")

    all_inferred_samples = []
    desc = 'Infering, num nan %d'
    all_sample_properties = []
    num_nan = 0
    with tqdm(enumerate(infer_dataloader), desc=desc % 0, total=len(infer_dataloader)) as bar:
        for batch_i, clean_sample in bar:
            if from_random_noise:
                sample = scheduler.gen_random_sample(clean_sample)
            else:
                sample, _ = scheduler.add_noise(
                    clean_sample,
                    torch.ones((clean_sample.get_num_atoms(),)).to(clean_sample.get_positions().device) * scheduler.t_max)
            sample_properties = [{key: val.cpu().item() for key, val in s.properties.items()} for s in sample.samples]

            sample_copy = sample.to(sample.get_positions().device)
            sample_copy.set_init_r_cut(init_r_cut)

            trajectory = scheduler.denoise(sample_copy, n_steps,
                                           con_weight=con_weight,
                                           non_stochastic=non_stochastic,
                                           hmc_n_iterations=hmc_n_iterations,
                                           hmc_n_steps=hmc_n_steps,
                                           hmc_dt=hmc_dt,
                                           hmc_range=hmc_range,
                                           log_file=hmc_logging_path)

            if torch.isnan(trajectory[-1].get_positions()).any():
                continue

            # save trajectories if required
            if save_name is not None and enable_trajectories and (save_traj_batches == 'all' or batch_i in save_traj_batches):
                ase_traj = [[at.to_ase_atoms() for at in batch.samples]
                            for batch in trajectory]
                for batch in ase_traj:
                    for at in batch:
                        at.wrap()
                for i_sample in range(sample.get_batch_size()):
                    traj = [s[i_sample] for s in ase_traj]
                    write_ase(os.path.join(save_dir, f'traj-{len(all_inferred_samples)+i_sample:05d}.extxyz'), traj)

            # Restart denoising for charge unbalanced samples if required
            final_sample = trajectory[-1]
            if max_restarts > 0:
                final_sample = restart_unbalanced_samples(final_sample, max_restarts, from_random_noise=False)
            if max_repeats > 0:
                final_sample = restart_unbalanced_samples(final_sample, max_repeats, from_random_noise=True)

            final_sample_ase = [at.to_ase_atoms() for at in final_sample.samples]
            for at in final_sample_ase:
                at.wrap()

            if save_name is not None:
                file_path = os.path.join(save_dir, 'inferred.extxyz')
                prop_path = os.path.join(save_dir, 'given_properties.json')
                if batch_i == 0:
                    write_ase(file_path, final_sample_ase, format='extxyz')
                else:
                    write_ase(file_path, final_sample_ase, format='extxyz', append=True)
                all_sample_properties.extend(sample_properties)
                with open(prop_path, 'w') as f:
                    json.dump(all_sample_properties, f)

            # collect inferred samples
            all_inferred_samples.extend(final_sample_ase)


def save_model(model, save_name, uncompiled_model=None, epoch=None):
    """Save the trained model parameters to a cache file.

    Args:
        model (nn.Module): the neural network model to save.
        save_name (str): an indicator of the model's name, used as the directory for saving this model.
        epoch (int, optional): indication of the saved model's number of trained epoches.
    """
    os.makedirs(os.path.join(save_dirs['models'], save_name), exist_ok=True)
    model_path = os.path.join(save_dirs['models'], save_name, f'{epoch:05d}.pt' if epoch is not None else 'final.pt')
    if uncompiled_model:
        torch.save(uncompiled_model.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_model(model, save_name, epoch=None):
    """Load model parameters from a cache file.
    """
    model_name = f'{epoch:05d}.pt' if epoch is not None else 'final.pt'
    model_path = os.path.join(save_dirs['models'], save_name, model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    print(f"Loaded model from {model_path}")

