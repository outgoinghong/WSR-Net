import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import h5py
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
#matplotlib.rc("font", family='Microsoft YaHei')
from einops import rearrange

class Data:
    def __init__(self, dataset, device):
        super(Data, self).__init__()

        data_path = "Datasets/" + dataset + ".mat"


        data = sio.loadmat(data_path)
        self.Y = torch.from_numpy(data['Y'].T).to(device)
        self.A = torch.from_numpy(data['GT'].T).to(device)
        self.M = torch.from_numpy(data['S_GT'])
        print(self.M)
        self.M1 = torch.from_numpy(data['S_GT'])

    def get(self, typ):
        if typ == "hs_img":
            return self.Y.float()
        elif typ == "abd_map":
            return self.A.float()
        elif typ == "end_mem":
            return self.M
        elif typ == "init_weight":
            return self.M1

    def get_hsi_mean(self):
        hsi_mean = np.mean(self.Y, axis=1)
        if len(hsi_mean.shape) == 2:
            hsi_mean = np.mean(hsi_mean, axis=0)
        hsi_mean = np.repeat(hsi_mean, self.P).reshape(-1, self.P).T
        return hsi_mean

class HSI:
    def __init__(self, data, rows, cols, gt, abundance_gt):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows
        self.image = np.reshape(data, (self.rows, self.cols, self.bands))
        self.gt = gt
        self.abundance_gt = abundance_gt

    def array(self):
        """返回 像元*波段 的数据阵列（array)"""

        return np.reshape(self.image, (self.rows * self.cols, self.bands))


def load_HSI(path):
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')

    numpy_array = np.asarray(data['Y'], dtype=np.float32)  # Y是波段*像元
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()

    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None

    if 'S_GT' in data.keys():
        abundance_gt = np.asarray(data['S_GT'], dtype=np.float32)
    else:
        abundance_gt = None

    return HSI(numpy_array, n_rows, n_cols, gt, abundance_gt)


def numpy_MSE(y_true, y_pred):  # 错写成MSE了，实际上是算RMSE，将错就错了
    num_cols = y_pred.shape[0]
    num_rows = y_pred.shape[1]
    diff = y_true - y_pred
    squared_diff = np.square(diff)
    mse = squared_diff.sum() / (num_rows * num_cols)
    rmse = np.sqrt(mse)
    return rmse


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_MSE(a[:, :, i], agt[:, :, j])

        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance


def numpy_SAD(y_true, y_pred, epsilon=1e-8, max_value=1e6):
    """
    计算两个向量之间的光谱角距离（SAD），加入数值稳定化措施避免除以零或inf。

    参数:
    y_true: 真实向量
    y_pred: 预测向量
    epsilon: 用于避免除以零的小值
    max_value: 最大允许的数值，防止出现 inf

    返回:
    光谱角距离，以弧度表示
    """
    # 限制向量的最大值，防止出现溢出导致的 inf
    y_true = np.clip(y_true, -max_value, max_value)
    y_pred = np.clip(y_pred, -max_value, max_value)

    # 计算两个向量的范数
    norm_true = np.linalg.norm(y_true)
    norm_pred = np.linalg.norm(y_pred)

    # 打印范数，便于调试
    #print(f"y_true 的范数: {norm_true}")
    #print(f"y_pred 的范数: {norm_pred}")

    # 检查范数是否为零或无穷大，防止除以零或 inf
    if norm_true < epsilon or np.isinf(norm_true) or norm_pred < epsilon or np.isinf(norm_pred):
        print("警告: 向量长度异常，无法计算光谱角距离")
        return np.nan  # 返回 NaN 表示无法计算

    # 计算余弦相似度
    cos = y_pred.dot(y_true) / (norm_true * norm_pred)

    # 确保 cos 的值在 [-1, 1] 之间，避免数值误差
    cos = np.clip(cos, -1.0, 1.0)

    # 返回光谱角距离（以弧度表示）
    return np.arccos(cos)


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    SAD_matrix = np.zeros((num_endmembers, num_endmembers))
    SAD_index = np.zeros(num_endmembers).astype(int)
    SAD_endmember = np.zeros(num_endmembers)
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    e = endmembers.copy()
    egt = endmembersGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            SAD_matrix[i, j] = numpy_SAD(e[i, :], egt[j, :])

        SAD_index[i] = np.nanargmin(SAD_matrix[i, :])
        SAD_endmember[i] = np.nanmin(SAD_matrix[i, :])
        egt[SAD_index[i], :] = np.inf
    return SAD_index, SAD_endmember


import matplotlib.pyplot as plt
import numpy as np


def plotEndmembersAndGT(endmembers, endmembersGT, endmember_path, sadsave):
    num_endmembers = endmembers.shape[0]

    # 设置列数为 2，动态计算行数
    cols = 2
    rows = (num_endmembers + cols - 1) // cols  # 确保子图有足够的行数

    SAD_index, SAD_endmember = order_endmembers(endmembersGT, endmembers)
    fig, axes = plt.subplots(rows, cols, figsize=(9, 9))  # 使用 subplots 动态生成子图
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # 调整子图之间的间距
    plt.rcParams.update({'font.size': 15})

    title = "mSAD: " + np.array2string(SAD_endmember.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x}) + " radians"
    st = fig.suptitle(title, y=0.95)

    # 归一化处理
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    # 生成子图
    for i in range(num_endmembers):
        row, col = divmod(i, cols)
        ax = axes[row, col] if rows > 1 else axes[col]  # 根据行数来选择正确的轴
        ax.plot(endmembers[SAD_index[i], :], 'r', linewidth=1.0)
        ax.plot(endmembersGT[i, :], 'k', linewidth=1.0)
        ax.set_title(format(numpy_SAD(endmembers[SAD_index[i], :], endmembersGT[i, :]), '.3f'))
        ax.get_xaxis().set_visible(False)
        sadsave.append(numpy_SAD(endmembers[SAD_index[i], :], endmembersGT[i, :]))

    sadsave.append(SAD_endmember.mean())

    # 如果子图数量不为偶数，隐藏多余的空子图
    if num_endmembers % cols != 0:
        for j in range(num_endmembers, rows * cols):
            fig.delaxes(axes.flatten()[j])  # 删除多余的空子图

    plt.savefig(endmember_path + '.png')
    """plt.draw()
    plt.pause(0.1)
    plt.close()"""


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_MSE(a[:, :, i], agt[:, :, j])

        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance


def plotAbundancesSimple(abundances, abundanceGT, abundance_path, rmsesave):
    abundances = np.transpose(abundances, axes=[1, 0, 2])  # 把行列颠倒，第三维不动，因为得到的丰度图和参考丰度图是互为转置的，即行列颠倒的
    num_endmembers = abundances.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    abundance_index, MSE_abundance = order_abundance(abundanceGT, abundances)
    title = "RMSE: " + np.array2string(MSE_abundance.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x})
    cmap = 'jet'
    plt.figure(figsize=[10, 10])
    AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, abundance_index[i]], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.set_title(format(numpy_MSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i]), '.3f'))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        rmsesave.append(numpy_MSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i]))

    rmsesave.append(MSE_abundance.mean())
    plt.tight_layout()  # 用于自动调整子图参数，以便使所有子图适合整个图像区域，并尽可能地减少子图之间的重叠
    plt.rcParams.update({'font.size': 15})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    """plt.draw()
    plt.pause(0.1)
    plt.close()"""


def plotAbundancesGT(abundanceGT, abundance_path):
    num_endmembers = abundanceGT.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    title = 'example abu'
    cmap = 'jet'
    plt.figure(figsize=[10, 10])
    AA = np.sum(abundanceGT, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundanceGT[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()  # 用于自动调整子图参数，以便使所有子图适合整个图像区域，并尽可能地减少子图之间的重叠
    plt.rcParams.update({'font.size': 19})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    plt.draw()
    plt.pause(0.1)
    plt.close()


def reconstruct(S, A):
    S = np.reshape(S, (S.shape[0] * S.shape[1], S.shape[2]))
    reconstructed = np.matmul(S, A)
    return reconstructed


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, transform=None):
        self.img = img.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img

    def __len__(self):
        return 1


def pca(X, d):
    N = np.shape(X)[1]
    xMean = np.mean(X, axis=1, keepdims=True)
    XZeroMean = X - xMean
    [U, S, V] = np.linalg.svd((XZeroMean @ XZeroMean.T) / N)
    Ud = U[:, 0:d]
    return Ud


def hyperVca(M, q, data):
    '''
    M : [L,N]
    '''
    L, N = np.shape(M)

    rMean = np.mean(M, axis=1, keepdims=True)
    RZeroMean = M - rMean
    U, S, V = np.linalg.svd(RZeroMean @ RZeroMean.T / N)
    Ud = U[:, 0:q]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(M ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (q / L) * P_R) / (P_R - P_Rp)))
    snrEstimate = SNR
    SNRth = 18 + 10 * np.log(q)

    if SNR > SNRth or data == 'Urban4':
        d = q
        U, S, V = np.linalg.svd(M @ M.T / N)
        Ud = U[:, 0:d]
        Xd = Ud.T @ M
        u = np.mean(Xd, axis=1, keepdims=True)
        Y = Xd / np.sum(Xd * u, axis=0, keepdims=True)

    else:
        d = q - 1
        r_bar = np.mean(M.T, axis=0, keepdims=True).T
        Ud = pca(M, d)

        R_zeroMean = M - r_bar
        Xd = Ud.T @ R_zeroMean
        c = [np.linalg.norm(Xd[:, j], ord=2) for j in range(N)]
        c = np.array(c)
        c = np.max(c, axis=0, keepdims=True) @ np.ones([1, N])
        Y = np.concatenate([Xd, c.reshape(1, -1)])
    e_u = np.zeros([q, 1])
    e_u[q - 1, 0] = 1
    A = np.zeros([q, q])
    A[:, 0] = e_u[0]
    I = np.eye(q)
    k = np.zeros([N, 1])

    indicies = np.zeros([q, 1])
    for i in range(q):  # i=1:q
        w = np.random.random([q, 1])
        tmpNumerator = (I - A @ np.linalg.pinv(A)) @ w
        f = tmpNumerator / np.linalg.norm(tmpNumerator)

        v = f.T @ Y
        k = np.abs(v)

        k = np.argmax(k)
        A[:, i] = Y[:, k]
        indicies[i] = k

    indicies = indicies.astype('int')
    if (SNR > SNRth) or data == 'Urban4':
        U = Ud @ Xd[:, indicies.T[0]]
    else:
        U = Ud @ Xd[:, indicies.T[0]] + r_bar

    return U, indicies, snrEstimate


def SAD(y_true, y_pred):
    y_true2 = torch.nn.functional.normalize(y_true, dim=1)
    y_pred2 = torch.nn.functional.normalize(y_pred, dim=1)
    A = torch.mean(y_true2 * y_pred2)
    sad = torch.acos(A)
    return sad

class SADLoss(nn.Module):
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if len(y_pred.shape) > 2:
            y_true = y_true.reshape(-1, y_true.shape[-1])
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.view(-1, 1, y_true.shape[1])
        y_pred = y_pred.view(-1, 1, y_pred.shape[1])
        y_true_norm = torch.sqrt(torch.bmm(y_true, y_true.permute(0, 2, 1)))
        y_pred_norm = torch.sqrt(torch.bmm(y_pred, y_pred.permute(0, 2, 1)))
        summation = torch.bmm(y_pred, y_true.permute(0, 2, 1))
        angle = torch.acos(summation / (y_true_norm * y_pred_norm))
        sad = torch.mean(angle)
        return sad


class My_Loss(nn.Module):
    def __init__(self, weight_mse=1.0, weight_sad=1.0, weight_endm=0.001, weight_aban=1e-6):
        super(My_Loss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_sad = weight_sad
        self.weight_endm = weight_endm
        self.weight_aban = weight_aban
        self.SAD = SADLoss()
        self.MSE = nn.MSELoss()

    def forward(self, y_true, y_pred, endm=None, hsi_mean=None, pred_aban=None):
        loss = 0
        if self.weight_mse != 0:
            loss += self.weight_mse * self.MSE(y_true, y_pred)

        if 1 < len(y_pred.shape) < 5:
            y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1).transpose(1, 2)
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1).transpose(1, 2)
        elif len(y_pred.shape) >= 5:
            y_true = rearrange(y_true, 'n b c w h -> (n b) (w h) c')
            y_pred = rearrange(y_pred, 'n b c w h -> (n b) (w h) c')

        if self.weight_sad != 0:
            loss_sad = self.weight_sad * self.SAD(y_true, y_pred)
            loss += loss_sad
        if endm is not None and hsi_mean is not None and self.weight_endm != 0:
            loss += self.weight_endm * self.MSE(hsi_mean, endm)
        if pred_aban is not None:
            aban_norm = torch.norm(pred_aban, p=0.5, dim=1)
            loss_aban = self.weight_aban * aban_norm.mean()
            loss += loss_aban
        return loss