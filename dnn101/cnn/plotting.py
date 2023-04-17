import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


def montage_array(A, num_col=None, cmap='viridis', names=None):
    # assume A is a 3D tensor for now
    # A[i, :, :] is the i-th image

    assert A.ndim == 3, "Montage array only available for third-order tensors"
    A = A.permute(1, 2, 0)

    m1, m2, m3 = A.shape

    if num_col is None:
        num_col = math.ceil(math.sqrt(m3))

    num_row = math.ceil(m3 / num_col)

    C = torch.nan * torch.ones([m1 * num_row + num_row - 1, m2 * num_col + num_col - 1])

    cmap = mpl.cm.viridis
    cmap.set_bad('black', 0.)

    k = 0
    shift_p = 0
    for p in range(0, num_row):
        shift_q = 0
        for q in range(0, num_col):
            if k >= m3:
                # C[p * m1 + shift_p:(p + 1) * m1 + shift_p, q * m2 + shift_q:(q + 1) * m2 + shift_q] = torch.nan
                break
            C[p * m1 + shift_p:(p + 1) * m1 + shift_p, q * m2 + shift_q:(q + 1) * m2 + shift_q] = A[:, :, k]
            k += 1
            shift_q += 1
        shift_p += 1
        

    img = plt.imshow(C, cmap=cmap)

    # grid lines
    # for i in range(0, C.shape[0] + 1, m1):
    #     plt.hlines(y=i-0.5, xmin=-0.5, xmax=C.shape[0]-0.5, linewidth=8, color='w')
    # for i in range(0, C.shape[1] + 1, m2):
    #     plt.vlines(x=i-0.5, ymin=-0.5, ymax=C.shape[1]-0.5, linewidth=8, color='w')

    plt.axis('off')
    # cb = plt.colorbar()
    cb = plt.colorbar(img, fraction=0.046, pad=0.04)

    if names is not None:
        cb.set_ticks(torch.arange(0, len(names)))
        cb.set_ticklabels(names)

    return img


def plot_Conv2d_filters(net, n_filters=4, num_col=None):

    if isinstance(n_filters, int):
        n_filters = torch.arange(n_filters)
    n = len(n_filters)

    stored_filters = []
    if isinstance(net, torch.nn.Conv2d):
        stored_filters.append(net.weight.detach())
    else:
        for p in net.children():
            if isinstance(p, torch.nn.Conv2d):
                stored_filters.append(p.weight.detach())

    for i in range(len(stored_filters)):
        plt.subplot(1, len(stored_filters), i + 1)
        w = stored_filters[i]
        w = w.view(-1, w.shape[2], w.shape[3])
        montage_array(w[n_filters[:min(n, w.shape[0])]], num_col=num_col)

    return stored_filters


def plot_CNN_features(net, x, n_features=4, num_col=None):
    # x is single image 1 x C x H x N
    if isinstance(n_features, int):
        n_features = torch.arange(n_features)
    n = len(n_features)
    N, C, H, W = x.shape


    stored_features = [x.detach()]

    for p in net.children():
        x = p(x)
        if isinstance(p, torch.nn.Conv2d):
            stored_features.append(x.detach())

    for i in range(len(stored_features)):
        plt.subplot(1, len(stored_features), i + 1)
        x = stored_features[i]
        x = x.view(-1, x.shape[2], x.shape[3])
        montage_array(x[:min(n, x.shape[0])], num_col=num_col)

    return stored_features


if __name__ == "__main__":
    layer = torch.nn.Conv2d(5, 7, 11, 1)
    # plot_Conv2d_filters(layer, n_filters=10, num_col=5)
    # plt.show()

    net = torch.nn.Sequential(torch.nn.Conv2d(5, 7, 3, 1), torch.nn.Conv2d(7, 2, 5, 2))
    # plot_Conv2d_filters(net)
    # plt.show()

    x = torch.randn(1, 5, 28, 28)
    plot_CNN_features(net, x)
    plt.show()
