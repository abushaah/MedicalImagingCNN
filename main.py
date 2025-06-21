from process import process
from transforms import transforms

# usage flow
if __name__ == '__main__':

    # 1. user input (for now, it is kaggle data set)
    params = process(in_dir="/kaggle/input")

    # testing
    print("process:")
    for k, v in params.items():
        print(f"{k}: {v}")

    # preprocess & show reasoning
    train_loader, validation_loader = transforms(pixdim=params['pixdim'], a_min=params['a_min'], a_max=params['a_max'], spatial_size=params['spatial_size'], batch_size=params['batch_size'], cache=params['mem_free_ram'], train_files=params['train_files'], validation_files=params['validation_files'])
    print(train_loader)
    print(validation_loader)
