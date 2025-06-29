import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import random
import argparse

class HeadProbeMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train HeadProbeMLP on cached transformer values")
    parser.add_argument("--data_path", type=str, default="data/Arco/Datasets/Probing/cached_values_2000_sin_correct_enc.selfatt[0].mab0_1.pt", help="Path to the .pt file")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for repeated training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size for MLP")
    parser.add_argument("--save_path", type=str, default="test_accuracy_seeds.png", help="Filename to save the accuracy/loss plot")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_probe(model, train_loader, val_loader, test_loader, epochs=1000, lr=1e-5, patience=30, device=None):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # === BASELINE ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs).squeeze()
            labels = labels.squeeze()
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = (val_correct / val_total) * 100
    history["train_loss"].append(val_loss)
    history["train_acc"].append(val_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)


    epoch_bar = tqdm(range(epochs), desc="Training")
    for epoch in epoch_bar:
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            labels = labels.squeeze()

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                labels = labels.squeeze()

                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        epoch_bar.set_postfix({
            "val_acc": f"{val_acc:.2f}%",
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
        })

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement in {patience} epochs)")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"Loading best epoch model from epoch: {best_epoch}")

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            labels = labels.squeeze()
            preds = (outputs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = (test_correct / test_total) * 100
    print(f"Test Accuracy: {test_acc}")
    return test_acc, history


def main(data_path, seeds=10, epochs=200, patience=200, lr=1e-4, batch_size=32, hidden_dim=32,
         save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(data_path)
    X_tensor, y_tensor = torch.load(data_path, weights_only=True)
    print(X_tensor.shape)

    y_np = y_tensor.squeeze().numpy()

    test_accuracies = []
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []

    for seed in range(seeds):
        set_seed(seed)
        print(f"\n\nTraining with seed: {seed}")

        train_idx, temp_idx = train_test_split(np.arange(len(y_np)), test_size=0.3, stratify=y_np, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, stratify=y_np[temp_idx], random_state=seed)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model = HeadProbeMLP(input_dim=X_tensor.shape[1], hidden_dim=hidden_dim)
        test_acc, history = train_probe(model, train_loader, val_loader, test_loader,
                                        epochs=epochs, patience=patience, lr=lr, device=device)

        test_accuracies.append(test_acc)
        all_train_losses.append(history["train_loss"])
        all_val_losses.append(history["val_loss"])
        all_train_accs.append(history["train_acc"])
        all_val_accs.append(history["val_acc"])

    print("Test Accuracies:", test_accuracies)

    all_train_losses = [run for run in all_train_losses]
    all_val_losses = [run for run in all_val_losses]
    all_train_accs = [run for run in all_train_accs]
    all_val_accs = [run for run in all_val_accs]

    min_len = min(len(run) for run in all_train_losses)
    all_train_losses = np.array([run[:min_len] for run in all_train_losses])
    all_val_losses = np.array([run[:min_len] for run in all_val_losses])
    all_train_accs = np.array([run[:min_len] for run in all_train_accs])
    all_val_accs = np.array([run[:min_len] for run in all_val_accs])

    epochs = range(min_len)
    if save_path is not None:
        big_size = 22
        small_size = 18
        purple = (0.6, 0.4, 0.8)
        blue = [0.1, 0.6, 0.8]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]

        ax.plot(epochs, all_val_losses.mean(0), color=purple, linewidth=3, label="Val Loss")
        ax.fill_between(epochs,
                        all_val_losses.mean(0) - all_val_losses.std(0),
                        all_val_losses.mean(0) + all_val_losses.std(0),
                        color=purple, alpha=0.2)

        ax.plot(epochs, all_train_losses.mean(0), color=blue, linewidth=3, label="Train Loss")
        ax.fill_between(epochs,
                        all_train_losses.mean(0) - all_train_losses.std(0),
                        all_train_losses.mean(0) + all_train_losses.std(0),
                        color=blue, alpha=0.2)

        ax.set_title("Loss", fontsize=big_size, weight='bold', color='black')
        ax.set_xlabel("Epoch", fontsize=small_size, weight='bold', color='black')
        ax.set_ylabel("Loss", fontsize=small_size, weight='bold', color='black')
        # ax.legend()
        ax.tick_params(axis='both', labelsize=small_size - 2, width=2, length=6, color='black', labelcolor='black')
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color("black")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- Accuracy Plot ---
        ax = axes[1]

        ax.plot(epochs, all_val_accs.mean(0), color=purple, linewidth=3, label="Val Acc")
        ax.fill_between(epochs,
                        all_val_accs.mean(0) - all_val_accs.std(0),
                        all_val_accs.mean(0) + all_val_accs.std(0),
                        color=purple, alpha=0.2)

        ax.plot(epochs, all_train_accs.mean(0), color=blue, linewidth=3, label="Train Acc")
        ax.fill_between(epochs,
                        all_train_accs.mean(0) - all_train_accs.std(0),
                        all_train_accs.mean(0) + all_train_accs.std(0),
                        color=blue, alpha=0.2)

        ax.set_title("Accuracy", fontsize=big_size, weight='bold', color='black')
        ax.set_xlabel("Epoch", fontsize=small_size, weight='bold', color='black')
        ax.set_ylabel("Accuracy", fontsize=small_size, weight='bold', color='black')
        ax.legend(fontsize=small_size)
        ax.tick_params(axis='both', labelsize=small_size - 2, width=2, length=6, color='black', labelcolor='black')
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color("black")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{save_path}.pdf")

    return test_accuracies


if __name__ == "__main__":
    args = parse_args()
    main(args)