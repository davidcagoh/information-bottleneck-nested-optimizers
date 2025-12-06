import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

from model import MLP
from mi_estimators import CLUB, CLUBForCategorical
from deep_momentum_optim.deep import DeepMomentum
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_fixed_subset(dataset, n=1000):
    idx = torch.randperm(len(dataset))[:n]
    return Subset(dataset, idx)

def estimate_mi(estimator, reps, labels, steps=50, lr=1e-3):
    opt = optim.Adam(estimator.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = estimator.learning_loss(reps, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    
    # Check if converged
    if abs(losses[-1] - losses[-10]) > 0.1:  # Still changing significantly
        print(f"WARNING: CLUB may not have converged (final loss: {losses[-1]:.4f})")
    
    with torch.no_grad():
        return estimator(reps, labels).mean().item()


def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds, _ = model(xb)
            pred = preds.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0

def main():
    set_seed(0)

    transform = transforms.ToTensor()
    train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
    # validation / test data
    test_data = datasets.MNIST('.', train=False, download=True, transform=transform)

    probe_set = get_fixed_subset(train_data, 500)
    probe_loader = DataLoader(probe_set, batch_size=256, shuffle=False)
    val_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # Hyperparameters (edit here to control runs without CLI)
    # Choose from: 'SGD', 'AdamW', 'GDM', 'DMGD'
    chosen_optim = 'AdamW'
    lr = 1e-3
    epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()

    safe_lr = str(lr).replace('.', 'p')

    # Build optimizer(s) according to choice
    manual_optim = False
    deep_opts = {}
    momentum_dict = {}
    beta = 0.9

    if chosen_optim == 'SGD':
        opt = optim.SGD(model.parameters(), lr=lr)
        optimizer_name = 'SGD'

    elif chosen_optim == 'AdamW':
        opt = optim.AdamW(model.parameters(), lr=lr)
        optimizer_name = 'AdamW'

    elif chosen_optim == 'GDM':
        manual_optim = True
        optimizer_name = 'GDM'
        for name, p in model.named_parameters():
            momentum_dict[name] = torch.zeros_like(p)

    elif chosen_optim == 'DMGD':
        manual_optim = True
        optimizer_name = 'DMGD'
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                deep_opts[name] = DeepMomentum().to(device)

    mi_x, mi_y = [], []

    # checkpointed accuracy tracking
    checkpoint_epochs = []
    train_acc_chkpts = []
    val_acc_chkpts = []

    # determine number of hidden layers from a single batch of the probe set
    with torch.no_grad():
        xb0, yb0 = next(iter(probe_loader))
        xb0, yb0 = xb0.to(device), yb0.to(device)
        _, acts0 = model(xb0)
    num_layers = len(acts0)

    # Warm-start estimators (persist across epochs to reuse weights)
    warm_est_y = {i: None for i in range(num_layers)}
    warm_est_x = {i: None for i in range(num_layers)}

    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")

        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds, _ = model(xb)
            loss = loss_fn(preds, yb)

            if not manual_optim:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    if chosen_optim == 'GDM':
                        for name, p in model.named_parameters():
                            if p.grad is None:
                                continue
                            momentum_dict[name] = beta * momentum_dict[name] + p.grad
                            p -= lr * momentum_dict[name]
                    elif chosen_optim == 'DMGD':
                        for name, module in model.named_modules():
                            if not isinstance(module, nn.Linear):
                                continue
                            optimizer_layer = deep_opts.get(name)
                            if optimizer_layer is None:
                                continue
                            for p in module.parameters():
                                if p.grad is None:
                                    continue
                                update = optimizer_layer(p.grad)
                                p.add_(update, alpha=-lr)

        model.eval()
        layer_repr = {i: [] for i in range(num_layers)}
        all_x, all_y = [], []

        with torch.no_grad():
            for xb, yb in probe_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, acts = model(xb)
                for i, a in enumerate(acts):
                    layer_repr[i].append(a)
                all_x.append(xb.view(xb.size(0), -1))
                all_y.append(yb)

        X = torch.cat(all_x)
        Y = torch.cat(all_y)

        mi_y_layers = []
        mi_x_layers = []


        for i in range(num_layers):
            R = torch.cat(layer_repr[i])
                
            # Initialize estimator if not existing, otherwise reuse and ensure it's on device
            if warm_est_y[i] is None:
                warm_est_y[i] = CLUBForCategorical(R.shape[1], Y.max().item()+1, hidden_size=128).to(device)
            else:
                warm_est_y[i] = warm_est_y[i].to(device)

            if warm_est_x[i] is None:
                warm_est_x[i] = CLUB(R.shape[1], X.shape[1], hidden_size=128).to(device)
            else:
                warm_est_x[i] = warm_est_x[i].to(device)

            # Detach representations so CLUB training does not backprop through the model
            mi_y_layers.append(estimate_mi(warm_est_y[i], R.detach(), Y))
            mi_x_layers.append(estimate_mi(warm_est_x[i], R.detach(), X.detach()))

        mi_y.append(mi_y_layers)
        mi_x.append(mi_x_layers)

        print(f"  Layer MI:")
        for i in range(num_layers):
            print(f"    Layer {i+1}: I(repr;Y)={mi_y_layers[i]:.4f}, I(repr;X)={mi_x_layers[i]:.4f}")

        # --- Save checkpoints for this epoch ---
        epoch_idx = ep + 1

        train_acc = compute_accuracy(model, probe_loader, device)
        val_acc = compute_accuracy(model, val_loader, device)
        checkpoint_epochs.append(epoch_idx)
        train_acc_chkpts.append(train_acc)
        val_acc_chkpts.append(val_acc)

        if epoch_idx % 10 ==0:
            torch.save(model.state_dict(), f"checkpoints/checkpoint_model_{epoch_idx}.pt")
            np.save(f"checkpoints/checkpoint_mi_x_{epoch_idx}.npy", np.array(mi_x))
            np.save(f"checkpoints/checkpoint_mi_y_{epoch_idx}.npy", np.array(mi_y))

            # --- Compute and save training/validation accuracy at this checkpoint ---


            # save per-checkpoint accuracy arrays and aggregated arrays
            np.save(f"checkpoints/checkpoint_acc_train_{epoch_idx}.npy", np.array(train_acc))
            np.save(f"checkpoints/checkpoint_acc_val_{epoch_idx}.npy", np.array(val_acc))
            np.save(f"checkpoints/checkpoint_acc_train_all.npy", np.array(train_acc_chkpts))
            np.save(f"checkpoints/checkpoint_acc_val_all.npy", np.array(val_acc_chkpts))
            np.save(f"checkpoints/checkpoint_epochs.npy", np.array(checkpoint_epochs))

            # --- Plot in mi_toy style for current progress and save per-epoch ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Trajectories (only plot epochs completed so far)
            ax = axes[0]
            epoch_colors = plt.cm.viridis(np.linspace(0, 1, epochs))
            for i in range(num_layers):
                # plot individual points colored by epoch
                for ep_idx in range(epoch_idx):
                    ax.plot(ep_idx, mi_x[ep_idx][i], marker='o', color=epoch_colors[ep_idx])
                    ax.plot(ep_idx, mi_y[ep_idx][i], marker='s', color=epoch_colors[ep_idx])

                # draw connecting lines for this layer across epochs completed so far
                xs_traj = list(range(epoch_idx))
                ys_x = [mi_x[e][i] for e in range(epoch_idx)]
                ys_y = [mi_y[e][i] for e in range(epoch_idx)]
                layer_color = plt.cm.tab10(i)
                # connect X trajectory (solid)
                ax.plot(xs_traj, ys_x, linestyle='-', color=layer_color, alpha=0.8)
                # connect Y trajectory with dashed line
                ax.plot(xs_traj, ys_y, linestyle='--', color=layer_color, alpha=0.8)
            # add legend for trajectories with composite handles (solid + dashed) per layer
            handles = []
            labels = []
            for i in range(num_layers):
                layer_color = plt.cm.tab10(i)
                line_solid = Line2D([0], [0], color=layer_color, lw=2, linestyle='-')
                line_dash = Line2D([0], [0], color=layer_color, lw=2, linestyle='--')
                handles.append((line_solid, line_dash))
                labels.append(f"Layer {i+1}")
            ax.legend(handles, labels, handler_map={tuple: HandlerTuple()})
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mutual Information (bits)")  # instead of just "Mutual Information"
            ax.set_title(f"Estimated MI ({optimizer_name}) — Epoch {epoch_idx}/{epochs} — lr={lr}")

            # Information plane scatter
            ax2 = axes[1]
            for i in range(num_layers):
                xs = [ep[i] for ep in mi_x]
                ys = [ep[i] for ep in mi_y]
                # scatter colored by epoch (no legend label)
                ax2.scatter(xs, ys, c=np.linspace(0,1,len(xs)), cmap='viridis', s=60)
                # connect the points in temporal order and attach label to the line
                layer_color = plt.cm.tab10(i)
                ax2.plot(xs, ys, linestyle='-', color=layer_color, alpha=0.8, label=f"Layer {i+1}")
                # highlight last point
                ax2.scatter(xs[-1], ys[-1], color='red', s=120, edgecolors='k')
            ax2.set_xlabel("I(repr; X)")
            ax2.set_ylabel("I(repr; Y)")
            ax2.axhline(y=3.32, color='red', linestyle=':', label='H(Y) = log₂(10)', alpha=0.5)
            ax2.set_title(f"Information Plane ({optimizer_name}) — Epoch {epoch_idx}/{epochs} — lr={lr}")
            ax2.legend()

            # Accuracy subplot (right)
            ax_acc = axes[2]
            if len(checkpoint_epochs) > 0:
                ax_acc.plot(checkpoint_epochs, train_acc_chkpts, '-o', label='Train Acc')
                ax_acc.plot(checkpoint_epochs, val_acc_chkpts, '-s', label='Val Acc')
            else:
                # fallback: plot current epoch accuracies as single point
                ax_acc.plot([epoch_idx], [train_acc], '-o', label='Train Acc')
                ax_acc.plot([epoch_idx], [val_acc], '-s', label='Val Acc')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_ylim(0, 1)
            ax_acc.set_title(f"Accuracy ({optimizer_name}) up to epoch {epoch_idx} — lr={lr}")
            ax_acc.legend()

            plt.tight_layout()
            plt.savefig(f"checkpoints/mi_acc_plot_{optimizer_name}_epoch{epoch_idx}_of{epochs}_lr{safe_lr}.png")
            plt.close(fig)

            # --- Plot in mi_toy style for current progress and save per-epoch ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Trajectories (only plot epochs completed so far)
            ax = axes[0]
            epoch_colors = plt.cm.viridis(np.linspace(0, 1, epochs))
            for i in range(num_layers):
                # plot individual points colored by epoch
                for ep_idx in range(epoch_idx):
                    ax.plot(ep_idx, mi_x[ep_idx][i], marker='o', color=epoch_colors[ep_idx])
                    ax.plot(ep_idx, mi_y[ep_idx][i], marker='s', color=epoch_colors[ep_idx])

                # draw connecting lines for this layer across epochs completed so far
                xs_traj = list(range(epoch_idx))
                ys_x = [mi_x[e][i] for e in range(epoch_idx)]
                ys_y = [mi_y[e][i] for e in range(epoch_idx)]
                layer_color = plt.cm.tab10(i)
                # connect X trajectory (solid)
                ax.plot(xs_traj, ys_x, linestyle='-', color=layer_color, alpha=0.8)
                # connect Y trajectory with dashed line
                ax.plot(xs_traj, ys_y, linestyle='--', color=layer_color, alpha=0.8)
            # add legend for trajectories with composite handles (solid + dashed) per layer
            handles = []
            labels = []
            for i in range(num_layers):
                layer_color = plt.cm.tab10(i)
                line_solid = Line2D([0], [0], color=layer_color, lw=2, linestyle='-')
                line_dash = Line2D([0], [0], color=layer_color, lw=2, linestyle='--')
                handles.append((line_solid, line_dash))
                labels.append(f"Layer {i+1}")
            ax.legend(handles, labels, handler_map={tuple: HandlerTuple()})
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mutual Information (bits)")  # instead of just "Mutual Information"
            ax.set_title(f"Estimated MI ({optimizer_name}) — Epoch {epoch_idx}/{epochs} — lr={lr}")

            # Information plane scatter
            ax2 = axes[1]
            for i in range(num_layers):
                xs = [ep[i] for ep in mi_x]
                ys = [ep[i] for ep in mi_y]
                # scatter colored by epoch (no legend label)
                ax2.scatter(xs, ys, c=np.linspace(0,1,len(xs)), cmap='viridis', s=60)
                # connect the points in temporal order and attach label to the line
                layer_color = plt.cm.tab10(i)
                ax2.plot(xs, ys, linestyle='-', color=layer_color, alpha=0.8, label=f"Layer {i+1}")
                # highlight last point
                ax2.scatter(xs[-1], ys[-1], color='red', s=120, edgecolors='k')
            ax2.set_xlabel("I(repr; X)")
            ax2.set_ylabel("I(repr; Y)")
            ax2.axhline(y=3.32, color='red', linestyle=':', label='H(Y) = log₂(10)', alpha=0.5)
            ax2.set_title(f"Information Plane ({optimizer_name}) — Epoch {epoch_idx}/{epochs} — lr={lr}")
            ax2.legend()

            plt.tight_layout()
            safe_lr = str(lr).replace('.', 'p')
            plt.savefig(f"checkpoints/mi_plot_{optimizer_name}_epoch{epoch_idx}_of{epochs}_lr{safe_lr}.png")
            plt.close(fig)

    # --- Plot in mi_toy style with accuracy as a third subplot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Trajectories (left)
    ax = axes[0]
    epoch_colors = plt.cm.viridis(np.linspace(0, 1, epochs))
    for i in range(num_layers):
        for ep_idx in range(epochs):
            ax.plot(ep_idx, mi_x[ep_idx][i], marker='o', color=epoch_colors[ep_idx])
            ax.plot(ep_idx, mi_y[ep_idx][i], marker='s', color=epoch_colors[ep_idx])
        # draw connecting lines (solid for X, dashed for Y)
        xs_traj = list(range(epochs))
        ys_x = [mi_x[e][i] for e in range(epochs)]
        ys_y = [mi_y[e][i] for e in range(epochs)]
        layer_color = plt.cm.tab10(i)
        ax.plot(xs_traj, ys_x, linestyle='-', color=layer_color, alpha=0.8)
        ax.plot(xs_traj, ys_y, linestyle='--', color=layer_color, alpha=0.8)
    # composite legend handles for final plot
    handles = []
    labels = []
    for i in range(num_layers):
        layer_color = plt.cm.tab10(i)
        line_solid = Line2D([0], [0], color=layer_color, lw=2, linestyle='-')
        line_dash = Line2D([0], [0], color=layer_color, lw=2, linestyle='--')
        handles.append((line_solid, line_dash))
        labels.append(f"Layer {i+1}")
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple()})
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mutual Information (bits)")  # instead of just "Mutual Information"
    ax.set_title(f"Estimated MI ({optimizer_name}) — {epochs} epochs — lr={lr}")

    # Information plane scatter (middle)
    ax2 = axes[1]
    for i in range(num_layers):
        xs = [ep[i] for ep in mi_x]
        ys = [ep[i] for ep in mi_y]
        # scatter colored by epoch (no legend label)
        ax2.scatter(xs, ys, c=np.linspace(0,1,epochs), cmap='viridis', s=60)
        # connect points and label the connecting line so legend uses its color
        layer_color = plt.cm.tab10(i)
        ax2.plot(xs, ys, linestyle='-', color=layer_color, alpha=0.8, label=f"Layer {i+1}")
        ax2.scatter(xs[-1], ys[-1], color='red', s=120, edgecolors='k')
    ax2.set_xlabel("I(repr; X)")
    ax2.set_ylabel("I(repr; Y)")
    ax2.axhline(y=3.32, color='red', linestyle=':', label='H(Y) = log₂(10)', alpha=0.5)
    ax2.set_title(f"Information Plane ({optimizer_name}) — {epochs} epochs — lr={lr}")
    ax2.legend()

    # Accuracy subplot (right)
    ax_acc = axes[2]
    if len(checkpoint_epochs) > 0:
        ax_acc.plot(checkpoint_epochs, train_acc_chkpts, '-o', label='Train Acc')
        ax_acc.plot(checkpoint_epochs, val_acc_chkpts, '-s', label='Val Acc')
    else:
        # fallback: plot final accuracies as a single point
        final_train = compute_accuracy(model, probe_loader, device)
        final_val = compute_accuracy(model, val_loader, device)
        ax_acc.plot([epochs], [final_train], '-o', label='Train Acc')
        ax_acc.plot([epochs], [final_val], '-s', label='Val Acc')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title(f"Accuracy ({optimizer_name}) — {epochs} epochs — lr={lr}")
    ax_acc.legend()

    plt.tight_layout()
    safe_lr = str(lr).replace('.', 'p')
    plt.savefig(f"output/mi_plot_{optimizer_name}_epochs{epochs}_lr{safe_lr}.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
