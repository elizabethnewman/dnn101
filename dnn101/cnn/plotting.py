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

    stored_filters = []
    if isinstance(net, torch.nn.Conv2d):
        n = min(len(n_filters), net.weight.shape[0] * net.weight.shape[1])
        stored_filters.append(net.weight.view(-1, net.weight.shape[2], net.weight.shape[3])[n_filters[:n]].detach())
    else:
        for p in net.children():
            if isinstance(p, torch.nn.Conv2d):
                n = min(len(n_filters), p.weight.shape[0] * p.weight.shape[1])
                stored_filters.append(p.weight.view(-1, p.weight.shape[2], p.weight.shape[3])[n_filters[:n]].detach())

    for i in range(len(stored_filters)):
        plt.subplot(1, len(stored_filters), i + 1)
        montage_array(stored_filters[i], num_col=num_col)

    return None


if __name__ == "__main__":
    layer = torch.nn.Conv2d(5, 7, 11, 1)
    plot_Conv2d_filters(layer, n_filters=10, num_col=5)
    plt.show()

    net = torch.nn.Sequential(torch.nn.Conv2d(5, 7, 11, 1), torch.nn.Conv2d(3, 1, 5, 2))
    plot_Conv2d_filters(net)
    plt.show()
