import json
import os
from argparse import ArgumentParser

import torch
import yaml
from torch.utils.data import DataLoader

import data
import pipeline
import utils
from data import save_dirs
from models.denoisers import egnn, egnn_deriv
from models.modules import losses
from models.modules.material_schedule import MaterialNoiseSchedule
from dotenv import load_dotenv


def main():
    setting, device, compile_model = parse_args_and_setting()
    model, orig_model, scheduler, loss_func, uncond_model = init_components(setting, device, compile_model)

    if setting.get('test', None) is not None and setting['test']['enabled']:
        test_dataloader = build_dataloader(setting['test']['data'], device)
        pipeline.test(model=model, scheduler=scheduler, dataloader=test_dataloader, tests=setting['test']['tests'])

    # Training process.
    start_epoch = setting['load']['epoch'] if setting['load']['enabled'] else 0
    if setting.get('train', None) is not None and setting['train']['enabled']:
        train_dataloader = build_dataloader(setting['train']['data'], device)
        pipeline.train(model=model, scheduler=scheduler, loss_func=loss_func, train_dataloader=train_dataloader,
                       orig_model=orig_model, start_epoch=start_epoch, **setting['train']['params'])

    # Load inferring dataset and dataloader.
    if setting.get('infer', None) is not None and setting['infer']['enabled']:
        infer_dataloader = build_dataloader(setting['infer'].get('data', setting['train']['data']), device)
        pipeline.infer(model=model, scheduler=scheduler, infer_dataloader=infer_dataloader, uncond_model=uncond_model,
                       **setting['infer']['params'])


def init_components(setting, device, compile_model):
    """Initializes the model, scheduler, and loss function components based on settings.
    
    Args:
        setting (dict): Dictionary containing model configurations.
            Expected structure:
            model:
              name: str  # Model type ('egnn_denoiser', 'egnn_deriv_denoiser', 'nequip_denoiser', or 'mace_denoiser')
              params: dict  # Model-specific parameters
            scheduler:
              params:
                sigma_max_pos: float  # Maximum position noise
                sigma_max_el: float  # Maximum element noise
                noise_schedule_el: str  # Noise schedule type
                t_min: float  # Minimum timestep
                t_max: float  # Maximum timestep
            loss:
              name: str  # Loss function type (e.g. 'epsilon_diff')
              params:
                norm_type: str  # Type of norm to use
                position_weight: float  # Weight for position loss
                element_weight: float  # Weight for element loss
            load:
              enabled: bool  # Whether to load pretrained model
              name: str  # Name of model to load
              epoch: int  # Epoch to load from
        device (str): Device to load the model to ('cuda:0' or 'cpu')
        compile_model (bool): Whether to compile the model using torch.compile()
    
    Returns:
        tuple:
            - model (nn.Module): The initialized model
            - orig_model (nn.Module): Original uncompiled model (if compile_model=True) or None
            - scheduler (MaterialNoiseSchedule): The noise scheduler
            - loss_func (nn.Module): The loss function
    """
    def _init_model(model_name, params):
        if model_name == 'egnn_denoiser':
            model = egnn.EgnnDenoiser(**params)
        elif model_name == 'egnn_deriv_denoiser':
            model = egnn_deriv.EgnnDerivDenoiser(**params)
        else:
            raise NotImplementedError(f'Unknown model name: {model_name}')
        return model

    # Initialize denoiser model.
    model = _init_model(setting['model']['name'], setting['model']['params'])

    # Load model parameters.
    if setting.get('load', None) is not None and setting['load']['enabled']:
        pipeline.load_model(model, setting['load']['name'], epoch=setting['load']['epoch'])
    model = model.to(device)
    
    if setting.get('load_uncond', None) is not None and setting['load_uncond']['enabled']:
        # Initialize the unconditioned model with same architecture but without properties.
        uncond_model_params = setting['model']['params'].copy()
        uncond_model_params['d_prop_embed'] = None
        uncond_model_params['properties'] = None
        uncond_model = _init_model(setting['model']['name'], uncond_model_params)
        pipeline.load_model(uncond_model, setting['load_uncond']['name'], epoch=setting['load_uncond']['epoch'])
        uncond_model = uncond_model.to(device)
    else:
        uncond_model = None

    n_model_params = sum([x.numel() for x in model.parameters()])
    n_uncond_model_params = sum([x.numel() for x in uncond_model.parameters()]) if uncond_model is not None else 0
    print(f'Model has {n_model_params} parameters')
    print(f'Unconditioned model has {n_uncond_model_params} parameters')

    if compile_model:
        # Backup the uncompiled model for parameter caching.
        orig_model = model
        model = torch.compile(orig_model)
    else:
        orig_model = None

    # Initialize noise scheduler.
    scheduler = MaterialNoiseSchedule(model, uncond_model=uncond_model, **setting['scheduler']['params'])

    # Initialize loss function.
    loss_name = setting['loss']['name']
    if loss_name == 'epsilon_diff':
        loss_func = losses.EpsilonDiff(**setting['loss']['params'])
    else:
        raise NotImplementedError(f'Unknown loss name: {loss_name}')

    return model, orig_model, scheduler, loss_func, uncond_model


def parse_args_and_setting():
    parser = ArgumentParser()
    parser.add_argument('-s', '--setting',
                        help='path to the setting file to use', type=str, required=True)
    parser.add_argument('-g', '--cuda',
                        help='index of the cuda (GPU) device to use', type=int, default=0)
    parser.add_argument('--compile', help='wether to compile the model with torch.compile',
                        action='store_true', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() and args.cuda is not None else 'cpu'
    if device != 'cpu':
        torch.set_float32_matmul_precision('high')  # for better performance on some gpus
    print('device: ' + device)

    compile_model = args.compile
    if compile_model:
        print('compiling model')

    # Load the setting file. It can be either JSON file (*.json) or YAML file (*.yml or *.yaml).
    with open(args.setting, 'r') as fp:
        if args.setting.endswith('.json'):
            setting = json.load(fp)
        elif args.setting.endswith('.yml') or args.setting.endswith('.yaml'):
            setting = yaml.safe_load(fp)
        else:
            raise NotImplementedError('The settings file has an unsupported format.')

    return setting, device, compile_model


def build_dataloader(data_setting, device):
    """Builds a DataLoader for training or inference based on the data configuration.
    
    Args:
        data_setting (dict): Dictionary containing dataset and dataloader configurations.
            Expected structure:
            dataset:
                target_density: float  # Target density for the material (e.g. 0.06)
                source:
                    name: str  # Source format (e.g. 'extxyz')
                    params: dict  # Source-specific parameters
            dataloader:
                batch_size: int  # Number of samples per batch
                num_workers: int  # Number of worker processes
                shuffle: bool  # Whether to shuffle the data
            collate: dict  # Optional collate function parameters
        device (str): Device to load the data to ('cuda:0' or 'cpu')
    
    Returns:
        DataLoader: PyTorch DataLoader configured with the specified dataset and parameters
    """
    dataset = data.MaterialDataset(**data_setting['dataset'])
    dataloader = DataLoader(dataset=dataset,
                            collate_fn=data.MaterialCollateFn(device, **data_setting.get('collate', {})),
                            **data_setting['dataloader'])
    return dataloader


if __name__ == '__main__':
    load_dotenv()
    main()
