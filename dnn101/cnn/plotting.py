import torch
import matplotlib.pyplot as plt
import math


def montage_array(A, num_col=None, cmap='viridis', names=None):
    # assume A is a 3D tensor for now
    # A[i, :, :] is the i-th image

    # assert A.ndim == 3, "Montage array only available for third-order tensors"
    A = A.permute(2, 3, 1, 0)

    m1, m2, _, m3 = A.shape

    if num_col is None:
        num_col = math.ceil(math.sqrt(m3))

    num_row = math.ceil(m3 / num_col)

    C = torch.zeros([m1 * num_row, m2 * num_col, A.shape[2]])

    k = 0
    for p in range(0, C.shape[0], m1):
        for q in range(0, C.shape[1], m2):
            if k >= m3:
                C[p:p + m1, q:q + m2] = torch.nan
                break
            C[p:p + m1, q:q + m2] = A[:, :, :, k]
            k += 1

    img = plt.imshow(C, cmap=cmap)
    plt.axis('off')
    # cb = plt.colorbar()
    cb = plt.colorbar(img, fraction=0.046, pad=0.04)

    if names is not None:
        cb.set_ticks(torch.arange(0, len(names)))
        cb.set_ticklabels(names)

    return img


def plot_convolution_filters(net, n_filters=4):

    if isinstance(n_filters, int):
        n_filters = torch.arange(n_filters)

    stored_filters = []
    for p in net.children():
        if isinstance(p, torch.nn.Conv2d):
            n = min(len(n_filters), p.weight.shape[0] * p.weight.shape[1])
            stored_filters.append(p.weight.view(-1, p.weight.shape[2], p.weight.shape[3])[n_filters[:n]].detach())

    for i in range(len(stored_filters)):
        plt.subplot(1, len(stored_filters), i + 1)
        montage_array(stored_filters[i])

    return None
