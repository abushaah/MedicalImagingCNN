from process import process
from transforms import transforms
from train import train

import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

if __name__ == '__main__':
    
    # 1. user input (for now, it is kaggle data set)
    params = prepare_and_configure(in_dir="/kaggle/input")
    
    # show output
    # print("prepare_and_configure:")
    # for k, v in params.items():
    #     print(f"{k}: {v}")
    
    # 2. preprocess
    train_loader, validation_loader = preprocess(pixdim=params['pixdim'], a_min=params['a_min'], a_max=params['a_max'], spatial_size=params['spatial_size'], batch_size=params['batch_size'], cache=params['mem_free_ram'], train_files=params['train_files'], validation_files=params['validation_files'])
    # print(train_loader)
    # print(validation_loader)

    # 3. build U-net
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2, 2),
        num_res_units=1,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # 4. train
    train(model, loss_function, optimizer, dice_metric)

    # 5. test
