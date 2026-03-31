# Example usage:
#   python make_sentence_embeddings.py data/train.tsv data/dev.tsv data/test.tsv \
#       --model embeddings.model \
#       --output_dir sentence_embeddings/

import argparse
import os
import re
import numpy as np
import pandas as pd
from gensim.models import FastText

def tokenize(text):
    """
    Split a Chinese sentence into character-level tokens.
    Chinese chars each become their own token; Latin/digit runs get lowercased
    and kept as one token. Punctuation and whitespace are dropped.
    """
    tokens = []
    for match in re.finditer(r'[\u4e00-\u9fff\u3400-\u4dbf]|[A-Za-z0-9]+', text):
        token = match.group()
        if token.isascii():
            tokens.append(token.lower())
        else:
            tokens.append(token)
    return tokens


# Sentence embedding

def embed_sentence(tokens, model, dim):

    if not tokens:
        # shouldn't really happen, but better safe than sorry
        return np.zeros(dim, dtype=np.float32)

    # Stack all character vectors into a matrix, then take the column-wise mean
    vectors = np.array([model.wv[token] for token in tokens], dtype=np.float32)
    return vectors.mean(axis=0)


#Argument parsing

def get_args():
    parser = argparse.ArgumentParser(
        description="Convert TSV dataset files into sentence-level FastText embeddings."
    )
    parser.add_argument(
        'tsv_files', nargs='+', metavar='TSV_FILE',
        help="TSV file(s) to embed. Usually train.tsv dev.tsv and test.tsv."
    )
    parser.add_argument(
        '--model', required=True,
        help="Path to the trained FastText model from train_embeddings.py."
    )
    parser.add_argument(
        '--output_dir', default='sentence_embeddings',
        help="Folder to write the .npz embedding files into. "
             "Will be created if it doesn't exist. (default: sentence_embeddings/)"
    )
    return parser.parse_args()


#Main

def main():
    args = get_args()

    # Make the output folder if it doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading FastText model from: {args.model}")
    model = FastText.load(args.model)
    dim = model.vector_size
    print(f"Model loaded. Each sentence will become a {dim}-dimensional vector.\n")

    for tsv_path in args.tsv_files:
        print(f"Processing: {tsv_path}")
        df = pd.read_csv(tsv_path, sep='\t')

        # The SIB-200 TSVs have columns 'text' and 'category'
        text_col  = 'text'     if 'text'     in df.columns else df.columns[-1]
        label_col = 'category' if 'category' in df.columns else df.columns[1]

        embeddings = []
        labels = []
        indices = []

        for row_idx, row in df.iterrows():
            tokens = tokenize(str(row[text_col]))
            emb = embed_sentence(tokens, model, dim)
            embeddings.append(emb)
            labels.append(str(row[label_col]))
            indices.append(row_idx)

        # Convert to numpy arrays for easy saving/loading
        embeddings = np.array(embeddings, dtype=np.float32)  # shape: (N, dim)
        labels = np.array(labels, dtype=object)              # shape: (N,)
        indices = np.array(indices, dtype=np.int64)          # shape: (N,)

        # Name the output file after the input (e.g. train.tsv -> train_embeddings.npz)
        base_name = os.path.splitext(os.path.basename(tsv_path))[0]
        out_path = os.path.join(args.output_dir, f"{base_name}_embeddings.npz")

        np.savez(out_path, embeddings=embeddings, labels=labels, indices=indices)

        print(f"  -> Saved {len(embeddings):,} sentence embeddings to: {out_path}")
        print(f"     Shape: {embeddings.shape}")
        print(f"     Label counts:")
        for label, count in pd.Series(labels).value_counts().items():
            print(f"       {label}: {count}")
        print()

    print("All done!")


if __name__ == '__main__':
    main()