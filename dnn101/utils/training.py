import torch


def evaluate(net, loss, data, device='cpu'):
    if isinstance(data, tuple):
        return evaluate_data(net, loss, data[0], data[1], device=device)
    else:
        return evaluate_dataloader(net, loss, data, device=device)


def train(net, loss, data, optimizer, batch_size=5, device='cpu'):

    if isinstance(data, tuple):
        return train_data(net, loss, data[0], data[1], optimizer, batch_size=batch_size, device=device)
    else:
        return train_dataloader(net, loss, data, optimizer, device=device)


def evaluate_data(net, loss, x, y, device='cpu'):
    with torch.no_grad():
        y_hat = net(x.to(device))
        phi = loss(y_hat, y.to(device))
        acc = y_hat.argmax(dim=1).eq(y.view(-1)).sum()

    return phi.item(), 100 * (acc / x.shape[0])


def train_data(net, loss, x, y, optimizer, scheduler=None, batch_size=5, device='cpu'):
    n_samples = x.shape[0]
    shuffle_idx = torch.randperm(n_samples)
    n_batch = n_samples // batch_size

    running_loss = 0.0
    running_acc = 0.0
    for i in range(n_batch):
        # select batch
        idx = shuffle_idx[i * batch_size:(i + 1) * batch_size]
        xb, yb = x[idx].to(device), y[idx].to(device)

        # zero out gradients
        optimizer.zero_grad()

        # forward propagate
        yb_hat = net(xb)

        # evaluate (with average loss)
        phi = loss(yb_hat, yb)
        running_loss += batch_size * phi.item()
        running_acc += yb_hat.argmax(dim=1).eq(yb.view(-1)).sum()

        # backward propagate (with automatic differentiation)
        phi.backward()

        # update (with optimizer rule)
        optimizer.step()

        if scheduler is not None:
            # adjust learning rate
            scheduler.step()

    return running_loss / (n_batch * batch_size), 100 * (running_acc / (n_batch * batch_size))


def evaluate_dataloader(net, loss, data_loader, device='cpu'):

    with torch.no_grad():
        phi, acc = torch.zeros(1, device=device), torch.zeros(1, device=device)
        n_samples, batch_size = 0, data_loader.batch_size
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_hat = net(xb)
            phi += batch_size * loss(yb_hat, yb)
            acc += yb_hat.argmax(dim=1).eq(yb.view(-1)).sum()
            n_samples += batch_size

    return phi.item() / n_samples, 100 * (acc.item() / n_samples)


def train_dataloader(net, loss, data_loader, optimizer, scheduler=None, device='cpu'):
    n_samples = 0
    batch_size = data_loader.batch_size
    running_loss = 0.0
    running_acc = 0.0
    for xb, yb in data_loader:
        # select batch
        xb, yb = xb.to(device), yb.to(device)
        n_samples += batch_size

        # zero out gradients
        optimizer.zero_grad()

        # forward propagate
        yb_hat = net(xb)

        # evaluate (with average loss)
        phi = loss(yb_hat, yb)
        running_loss += batch_size * phi.item()
        running_acc += yb_hat.argmax(dim=1).eq(yb.view(-1)).sum()

        # backward propagate (with automatic differentiation)
        phi.backward()

        # update (with optimizer rule)
        optimizer.step()

        if scheduler is not None:
            # adjust learning rate
            scheduler.step()

    return running_loss / n_samples, 100 * (running_acc / n_samples)
