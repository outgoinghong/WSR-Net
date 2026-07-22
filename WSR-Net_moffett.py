import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.io as sio
import torchvision.transforms as transforms
from utility import load_HSI, hyperVca, load_data, reconstruction_SADloss, Data, My_Loss
from utility import plotAbundancesGT, plotAbundancesSimple, plotEndmembersAndGT, reconstruct

from model import WAANet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


CONFIGS = {
    'moffett': {'seed': 1, 'lr': 4e-3, 'step_size': 35, 'gamma': 0.5, 'weight_decay': 8e-4, 'beta': 1e-4, 'a': 1,
                'epochs': 500},
}

dataset_name = "Samson"
cfg = CONFIGS[dataset_name]
set_seed(cfg['seed'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()


hsi = load_HSI(f"Datasets/{dataset_name}.mat")
data = hsi.array()
P = hsi.gt.shape[0]  
col, line = hsi.cols, hsi.rows
L = data.shape[1]  
batch_size = 1
num_runs = 1


abundance_GT = torch.from_numpy(hsi.abundance_gt).reshape(col * line, P).permute(1, 0).reshape(P, col, line)

original_HSI = torch.from_numpy(data.reshape(col, line, L)).permute(2, 0, 1).float()


endmembers, _, _ = hyperVca(data.T, P, dataset_name)
endmember_init = torch.from_numpy(endmembers).unsqueeze(2).unsqueeze(3).float()

# DataLoader
train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

output_path = 'Results'
method_name = 'WAA-Net'
base_folder = os.path.join(output_path, method_name, dataset_name)

mat_folder = os.path.join(base_folder, 'mat')
endmember_folder = os.path.join(base_folder, 'endmember')
abundance_folder = os.path.join(base_folder, 'abundance')

os.makedirs(mat_folder, exist_ok=True)
os.makedirs(endmember_folder, exist_ok=True)
os.makedirs(abundance_folder, exist_ok=True)

end_list, abu_list, r_list = [], [], []
loss_mse_func = nn.MSELoss(reduction='mean')

for run in range(1, num_runs + 1):
    print(f'Start training! run: {run}')

    net = WAANet(P=P, L=L).to(device)


    model_dict = net.state_dict()


    model_dict["decoder.base_E"] = endmember_init.squeeze()

    net.load_state_dict(model_dict)

    print("[✔] VCA 先验端元已成功注入为 PLMM 基准端元 (base_E)！")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])

    net.train()
    for epoch in range(cfg['epochs']):
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            en_abundance, re, delta_E = net(x)

            loss_sad = cfg['a'] * reconstruction_SADloss(x, re)
            loss_mse = cfg['beta'] * loss_mse_func(re, x)

            loss_pert = 0.1 * torch.mean(torch.norm(delta_E, p=2, dim=1))  
            total_loss = loss_sad + loss_mse + loss_pert

            total_loss.backward()
            optimizer.step()

        scheduler.step()
        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | Total Loss: {total_loss.item():.4f} "
                f"(SAD: {loss_sad.item():.4f}, MSE: {loss_mse.item():.4e}, PERT: {loss_pert.item():.4e})")

    net.eval()
    with torch.no_grad():
        test_input = original_HSI.unsqueeze(0).to(device)
        en_abundance, _, _ = net(test_input)

    en_abundance = en_abundance.squeeze().permute(1, 2, 0).cpu().numpy()  # (col, line, P)
    abundance_GT_np = abundance_GT.permute(1, 2, 0).cpu().numpy()

    endmember_hat = net.extract_base_endmembers().cpu().numpy().T
    y_hat = reconstruct(en_abundance, endmember_hat)
    RE = np.sqrt(np.mean(np.mean((y_hat - data) ** 2, axis=1)))
    r_list.append(RE)
    run_prefix = f"{dataset_name}_run{run}"
    sio.savemat(os.path.join(mat_folder, f'{method_name}_run{run}.mat'), {
        'A': en_abundance, 'E': endmember_hat
    })

    plotAbundancesSimple(en_abundance, abundance_GT_np, os.path.join(abundance_folder, run_prefix), abu_list)
    plotEndmembersAndGT(endmember_hat, hsi.gt, os.path.join(endmember_folder, run_prefix), end_list)

    print('-' * 70)

end_time = time.time()
print(f'total time: {end_time - start_time:.2f} s')

end_arr = np.reshape(end_list, (-1, P + 1))
abu_arr = np.reshape(abu_list, (-1, P + 1))

pd.DataFrame(end_arr).to_csv(os.path.join(base_folder, f'{dataset_name}各端元SAD及mSAD运行结果.csv'))
pd.DataFrame(abu_arr).to_csv(os.path.join(base_folder, f'{dataset_name}各丰度图RMSE及mRMSE运行结果.csv'))
pd.DataFrame(r_list).to_csv(os.path.join(base_folder, f'{dataset_name}重构误差RE运行结果.csv'))

plotAbundancesGT(hsi.abundance_gt, os.path.join(base_folder, f'{dataset_name}参照丰度图'))
