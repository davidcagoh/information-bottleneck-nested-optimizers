import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

from model import MLP
from mi_estimators import CLUB, CLUBForCategorical, TrueCategoricalMI, gaussian_mi_bits
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

def main(chosen_optim: str, epochs: int): # <--- ADD ARGUMENTS HERE
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
    # chosen_optim = 'SGD'
    lr = 1e-3
    # epochs = 3999
    print(f"Using optimizer: {chosen_optim}, Epochs: {epochs}")

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
    # Add separate optimizers for each layer's categorical MI estimator
    mi_y_opts = {i: None for i in range(num_layers)}
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
            R = torch.cat(layer_repr[i])          # [N, d_T]
            R = R.detach()

            # ----- initialize estimator + optimizer -----
            if warm_est_y[i] is None:
                warm_est_y[i] = TrueCategoricalMI(
                    R.shape[1], 
                    Y.max().item() + 1
                ).to(device)
                mi_y_opts[i] = torch.optim.Adam(warm_est_y[i].parameters(), lr=1e-3)
            else:
                warm_est_y[i] = warm_est_y[i].to(device)

            # ----- train estimator -----
            warm_est_y[i].train()
            for _ in range(10):
                idx = torch.randperm(R.shape[0])[:1024]
                batch_R = R[idx]
                batch_Y = Y[idx]

                loss = warm_est_y[i].learning_loss(batch_R, batch_Y)
                mi_y_opts[i].zero_grad()
                loss.backward()
                mi_y_opts[i].step()

            # ----- evaluate I(T;Y) -----
            warm_est_y[i].eval()
            I_ty = warm_est_y[i](R, Y.detach())   # bits
            mi_y_layers.append(I_ty)

            # ----- compute Gaussian I(T;X) -----
            I_tx = gaussian_mi_bits(
                R, 
                sigma=0.1, 
                eps=1e-6, 
                max_samples=3000,
                device=device
            )
            mi_x_layers.append(I_tx)

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
        print(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")  
        
    # --- Create figure ---
    # Change to (2, 2) for a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Adjust figsize for better viewing
    
    # Flatten the axes array for easier indexing: axes[0, 0] becomes axes[0], etc.
    axes = axes.flatten() 
    
    epoch_colors = plt.cm.viridis(np.linspace(0, 1, epochs))
    layer_colors = [plt.cm.tab10(i % 10) for i in range(num_layers)]

    # ============================================================
    # 1. I(T;X) vs EPOCH (Top-Left Panel: axes[0])
    # ============================================================
    ax_itx = axes[0] 

    for i in range(num_layers):
        col = layer_colors[i]

        # Extract per-layer sequences across epochs
        xs = np.arange(epochs)
        ys_x = [mi_x[e][i] for e in range(epochs)]

        # Plot lines
        ax_itx.plot(xs, ys_x, '-', color=col, lw=2, alpha=0.9, label=f"Layer {i+1}")

        # Overlay colored epoch markers (epoch → color)
        for e in range(epochs):
            ax_itx.plot(xs[e], ys_x[e], 'o', color=epoch_colors[e])

    ax_itx.legend(title="Layers")
    ax_itx.set_xlabel("Epoch")
    ax_itx.set_ylabel("I(T;X) (bits)")
    ax_itx.set_title(f"I(T;X) Trajectories — {optimizer_name} (lr={lr})")


    # ============================================================
    # 2. I(T;Y) vs EPOCH (Top-Right Panel: axes[1])
    # ============================================================
    ax_ity = axes[1] 

    for i in range(num_layers):
        col = layer_colors[i]

        # Extract per-layer sequences across epochs
        xs = np.arange(epochs)
        ys_y = [mi_y[e][i] for e in range(epochs)]

        # Plot lines
        ax_ity.plot(xs, ys_y, '--', color=col, lw=2, alpha=0.9, label=f"Layer {i+1}")

        # Overlay colored epoch markers (epoch → color)
        for e in range(epochs):
            ax_ity.plot(xs[e], ys_y[e], 's', color=epoch_colors[e])

    ax_ity.legend(title="Layers")
    ax_ity.set_xlabel("Epoch")
    ax_ity.set_ylabel("I(T;Y) (bits)")
    ax_ity.set_title(f"I(T;Y) Trajectories")
    
    
    # ============================================================
    # 3. INFORMATION PLANE (Bottom-Left Panel: axes[2])
    # ============================================================
    ax2 = axes[2] 

    for i in range(num_layers):
        col = layer_colors[i]

        xs = [ep[i] for ep in mi_x]
        ys = [ep[i] for ep in mi_y]

        # Epoch-colored scatter
        sc = ax2.scatter(xs, ys, c=np.linspace(0,1,epochs),
                        cmap='viridis', s=60)

        # Connect trajectory
        ax2.plot(xs, ys, '-', color=col, lw=2, alpha=0.8, label=f"Layer {i+1}")

        # Mark final epoch
        ax2.scatter(xs[-1], ys[-1], color='red', s=120, edgecolors='black', zorder=5)

    # Note: np.log2(10) is the maximum possible I(T;Y) for 10 classes
    ax2.axhline(y=np.log2(10), color='red', linestyle=':', alpha=0.4,
                label="$H(Y)=\\log_2(10)$") 
    ax2.set_xlabel("I(T;X)")
    ax2.set_ylabel("I(T;Y)")
    ax2.set_title("Information Plane")
    ax2.legend()


    # ============================================================
    # 4. ACCURACY PLOT (Bottom-Right Panel: axes[3])
    # ============================================================
    ax3 = axes[3] 

    if len(checkpoint_epochs) > 0:
        ax3.plot(checkpoint_epochs, train_acc_chkpts, '-o', label='Train Acc')
        ax3.plot(checkpoint_epochs, val_acc_chkpts, '-s', label='Val Acc')
    else:
        # Assuming compute_accuracy, model, probe_loader, device, val_loader are defined
        final_train = compute_accuracy(model, probe_loader, device) 
        final_val = compute_accuracy(model, val_loader, device)
        ax3.plot([epochs], [final_train], 'o', label='Train Acc')
        ax3.plot([epochs], [final_val], 's', label='Val Acc')

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Accuracy")
    ax3.legend()

    # ============================================================
    plt.tight_layout()
    safe_lr = str(lr).replace(".", "p")
    plt.savefig(f"output/mi_4panel_{optimizer_name}_epochs{epochs}_lr{safe_lr}_2x2.png")
    plt.close(fig)
