
# Example (with dev set + training curve plot, for the bonus):
#   python train_classifier.py \
#       --train sentence_embeddings/train_embeddings.npz \
#       --dev   sentence_embeddings/dev_embeddings.npz \
#       --labels data/labels.txt \
#       --output model.pt \
#       --epochs 30 --plot training_curve.png

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Model definition

class TopicClassifier(nn.Module):
    """
    A simple feed-forward network for topic classification.

    We build the hidden layers dynamically based on the --hidden argument,
    so you can experiment with different sizes without changing the code.

    Each hidden layer is: Linear -> ReLU -> Dropout
    The final layer is just a Linear (no activation -- CrossEntropyLoss
    expects raw scores, not probabilities).
    """

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()

        layers = []
        prev_size = input_dim

        for hidden_size in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Final output layer: one score per class
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Helpers

def load_embeddings(path):
    """Load a .npz file and return (embeddings array, labels array)."""
    data = np.load(path, allow_pickle=True)
    return data['embeddings'].astype(np.float32), data['labels'].astype(str)


def build_label_map(labels_file):
    """
    Read labels.txt and return a dict mapping each category name to an integer.
    e.g. {'science & technology': 0, 'travel': 1, ...}
    """
    with open(labels_file) as f:
        categories = [line.strip() for line in f if line.strip()]
    return {cat: i for i, cat in enumerate(categories)}


def labels_to_ints(label_strings, label_map):
    """Convert an array of string labels to integer class indices."""
    return np.array([label_map[label] for label in label_strings], dtype=np.int64)


def compute_accuracy(logits, targets):
    """
    Given raw model output (logits) and the correct class indices,
    return the fraction of correct predictions.
    """
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


# Argument parsing

def get_args():
    parser = argparse.ArgumentParser(
        description="Train a feed-forward classifier for Chinese topic classification."
    )
    parser.add_argument(
        '--train', required=True,
        help="Path to train_embeddings.npz (output of make_sentence_embeddings.py)."
    )
    parser.add_argument(
        '--dev', default=None,
        help="(Optional) Path to dev_embeddings.npz. If provided, prints dev accuracy "
             "each epoch and can generate a training curve plot."
    )
    parser.add_argument(
        '--labels', required=True,
        help="Path to labels.txt listing all topic category names."
    )
    parser.add_argument(
        '--output', default='model.pt',
        help="Where to save the trained model checkpoint. (default: model.pt)"
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help="Number of training epochs. (default: 20)"
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help="Number of examples per mini-batch. (default: 64)"
    )
    parser.add_argument(
        '--hidden', type=int, nargs='+', default=[256, 128],
        help="Sizes of the hidden layers. E.g. --hidden 256 128 gives two layers. "
             "(default: 256 128)"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help="Learning rate for Adam optimizer. (default: 0.001)"
    )
    parser.add_argument(
        '--dropout', type=float, default=0.3,
        help="Dropout probability -- helps prevent overfitting. (default: 0.3)"
    )
    parser.add_argument(
        '--plot', default=None,
        help="(Optional, bonus) Save a training curve PNG to this path. "
             "Requires --dev to be set."
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility. (default: 42)"
    )
    return parser.parse_args()


#  Main

def main():
    args = get_args()

    # Seeds so results are reproducible across runs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Label Map
    label_map = build_label_map(args.labels)
    num_classes = len(label_map)
    print(f"Found {num_classes} categories: {', '.join(label_map.keys())}")

    # Load training
    X_train, y_train_raw = load_embeddings(args.train)
    y_train = labels_to_ints(y_train_raw, label_map)
    input_dim = X_train.shape[1]
    print(f"Training on {len(X_train):,} sentences with {input_dim}-dim embeddings.")

    # Wrap in PyTorch dataset and dataloader for batching + shuffling
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load dev data
    using_dev = args.dev is not None
    if using_dev:
        X_dev, y_dev_raw = load_embeddings(args.dev)
        y_dev = labels_to_ints(y_dev_raw, label_map)
        # Move dev data to device once -- we reuse it every epoch
        X_dev_tensor = torch.tensor(X_dev).to(device)
        y_dev_tensor = torch.tensor(y_dev).to(device)
        print(f"Dev set: {len(X_dev):,} sentences.")

    # Build the model
    model = TopicClassifier(input_dim, args.hidden, num_classes, args.dropout).to(device)
    print(f"\nModel architecture:\n{model}\n")

    # CrossEntropyLoss combines softmax + negative log-likelihood, which is
    # the standard setup for multi-class classification.
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loop
    # Keep track of stats for the plot at the end
    train_losses = []
    train_accs   = []
    dev_accs     = []

    for epoch in range(1, args.epochs + 1):
        model.train()  # turn on dropout

        epoch_loss  = 0.0
        epoch_acc   = 0.0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Standard PyTorch training step:
            optimizer.zero_grad()       # 1. clear old gradients
            logits = model(X_batch)     # 2. forward pass
            loss   = loss_fn(logits, y_batch)  # 3. compute loss
            loss.backward()             # 4. backprop
            optimizer.step()            # 5. update weights

            epoch_loss  += loss.item()
            epoch_acc   += compute_accuracy(logits, y_batch)
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_acc  = epoch_acc  / num_batches
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        if using_dev:
            # Evaluate on dev set 
            model.eval()
            with torch.no_grad():
                dev_logits = model(X_dev_tensor)
            dev_acc = compute_accuracy(dev_logits, y_dev_tensor)
            dev_accs.append(dev_acc)
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  train_acc={avg_acc:.4f}  dev_acc={dev_acc:.4f}")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  train_acc={avg_acc:.4f}")

    # Save the model
    label_inv = {v: k for k, v in label_map.items()}  # int -> str, for decoding predictions

    torch.save({
        'model_state': model.state_dict(),
        'input_dim':   input_dim,
        'hidden_dims': args.hidden,
        'num_classes': num_classes,
        'dropout':     args.dropout,
        'label_map':   label_map, 
        'label_inv':   label_inv, 
    }, args.output)

    print(f"\nModel saved to: {args.output}")

    # Training curve plot (bonus part 1)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg') 
            import matplotlib.pyplot as plt

            epoch_nums = range(1, args.epochs + 1)

            fig, ax1 = plt.subplots(figsize=(9, 5))

            # Left y-axis: loss
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='steelblue')
            ax1.plot(epoch_nums, train_losses, color='steelblue',
                     linewidth=2, label='Train Loss')
            ax1.tick_params(axis='y', labelcolor='steelblue')

            # Right y-axis: accuracy (shares the x-axis)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Accuracy', color='darkorange')
            ax2.plot(epoch_nums, train_accs, color='darkorange',
                     linestyle='--', linewidth=2, label='Train Acc')
            if using_dev:
                ax2.plot(epoch_nums, dev_accs, color='seagreen',
                         linestyle='-', linewidth=2, label='Dev Acc')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='y', labelcolor='darkorange')

            # Combine both legends into one box
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='center right')

            plt.title('Training Curve -- Chinese Topic Classifier')
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"Training curve saved to: {args.plot}")

        except Exception as e:
            print(f"Warning: couldn't save the plot. Error: {e}")


if __name__ == '__main__':
    main()