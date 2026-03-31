import argparse
import numpy as np
import torch
import torch.nn as nn

class TopicClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Helper functions

def load_embeddings(path):
    """Load a .npz embedding file, return (X array, y labels array)."""
    data = np.load(path, allow_pickle=True)
    return data['embeddings'].astype(np.float32), data['labels'].astype(str)


def print_confusion_matrix(matrix, class_names):

    num_classes = len(class_names)

    # Figure out column width so everything lines up nicely
    col_width = max(len(name) for name in class_names)
    col_width = max(col_width, 5)
    row_label_width = col_width + 2

    # Print column headers
    header = ' ' * row_label_width + '  '.join(f'{name:>{col_width}}' for name in class_names)
    print(header)
    print('-' * len(header))

    # Print each row
    for i, row_name in enumerate(class_names):
        row = f'{row_name:<{row_label_width}}'
        row += '  '.join(f'{matrix[i, j]:>{col_width}d}' for j in range(num_classes))
        print(row)


# Argument parsing

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Chinese topic classifier on the test set."
    )
    parser.add_argument(
        '--model', required=True,
        help="Path to the saved model checkpoint (.pt file from train_classifier.py)."
    )
    parser.add_argument(
        '--test', required=True,
        help="Path to test_embeddings.npz (from make_sentence_embeddings.py)."
    )
    return parser.parse_args()


# Main

def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reconstruct the model from the checkpoint
    # I saved everything needed to rebuild the model inside the .pt file.
    checkpoint  = torch.load(args.model, map_location=device)
    label_map   = checkpoint['label_map']
    label_inv   = checkpoint['label_inv']
    num_classes = checkpoint['num_classes']
    class_names = [label_inv[i] for i in range(num_classes)]

    model = TopicClassifier(
        input_dim   = checkpoint['input_dim'],
        hidden_dims = checkpoint['hidden_dims'],
        num_classes = num_classes,
        dropout     = checkpoint['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded from: {args.model}")

    # Load the test data
    X_test, y_test_strings = load_embeddings(args.test)
    y_true = np.array([label_map[label] for label in y_test_strings], dtype=np.int64)
    X_tensor = torch.tensor(X_test).to(device)

    # Run inference (no gradient tracking needed)
    with torch.no_grad():
        logits = model(X_tensor)
    y_pred = logits.argmax(dim=1).cpu().numpy()

    # Compute overall accuracy
    num_correct = (y_pred == y_true).sum()
    total       = len(y_true)
    accuracy    = num_correct / total

    # If the model just guessed randomly and uniformly, it would get 1/num_classes right.
    chance_level = 1.0 / num_classes
    above_chance = accuracy > chance_level

    print(f"\n{'=' * 55}")
    print(f"  Test samples : {total:,}")
    print(f"  Accuracy     : {accuracy:.4f}  ({num_correct}/{total} correct)")
    print(f"  Chance level : {chance_level:.4f}  (random guessing over {num_classes} classes)")
    print(f"  Above chance : {'YES' if above_chance else 'NO'}  "
          f"(difference: {accuracy - chance_level:+.4f})")
    print(f"{'=' * 55}\n")

    # Per-class accuracy
    print("Per-class accuracy:")
    print(f"  {'Category':<30}  Correct / Total   Acc")
    print(f"  {'-'*55}")
    for i, name in enumerate(class_names):
        is_this_class = (y_true == i)
        n_total   = is_this_class.sum()
        n_correct = ((y_pred == i) & is_this_class).sum()
        cls_acc   = n_correct / n_total if n_total > 0 else 0.0
        print(f"  {name:<30}  {n_correct:3d} / {n_total:3d}         {cls_acc:.3f}")

    # Confusion matrix
    print("\nConfusion Matrix  (rows = true label, columns = predicted label):\n")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    print_confusion_matrix(cm, class_names)

    # Most common mistakes 
    # Collect all off-diagonal cells (where the prediction was wrong) and sort by count
    print(f"\nTop confused pairs (true -> predicted):")
    mistakes = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                mistakes.append((cm[i, j], class_names[i], class_names[j]))
    mistakes.sort(reverse=True)

    for count, true_class, pred_class in mistakes[:10]:
        print(f"  {true_class:<30}  ->  {pred_class:<30}  ({count} times)")


if __name__ == '__main__':
    main()