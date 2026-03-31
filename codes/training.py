import argparse
import re
import pandas as pd
from gensim.models import FastText


def tokenize(text):
    """
    Split a Chinese sentence into a list of character-level tokens.
    """
    tokens = []
    # This regex matches either one Chinese character OR a run of latin/digit chars.
    for match in re.finditer(r'[\u4e00-\u9fff\u3400-\u4dbf]|[A-Za-z0-9]+', text):
        token = match.group()
        if token.isascii():
            tokens.append(token.lower())  # lowercase latin stuff like "gps", "covid19"
        else:
            tokens.append(token)  # one Chinese character = one token
    return tokens


def load_all_sentences(tsv_files):
    all_sentences = []
    for path in tsv_files:
        df = pd.read_csv(path, sep='\t')

        # The SIB-200 dataset has a column called 'text'. Grab it, or fall back
        # to the last column if the name is ever different.
        text_col = 'text' if 'text' in df.columns else df.columns[-1]

        for sentence in df[text_col].dropna():
            tokens = tokenize(str(sentence))
            if tokens:  # skip any rows that ended up empty after tokenizing
                all_sentences.append(tokens)

    print(f"Loaded {len(all_sentences):,} sentences from {len(tsv_files)} file(s).")
    return all_sentences


def get_args():
    parser = argparse.ArgumentParser(
        description="Train FastText character embeddings on the Chinese SIB-200 dataset."
    )

    parser.add_argument(
        'tsv_files', nargs='+', metavar='TSV_FILE',
        help="Path(s) to .tsv data files, e.g. train.tsv dev.tsv test.tsv"
    )
    parser.add_argument(
        '--dim', type=int, default=100,
        help="Size of each embedding vector. 100 is a solid default. (default: 100)"
    )
    parser.add_argument(
        '--output', default='embeddings.model',
        help="Filename to save the trained FastText model. (default: embeddings.model)"
    )
    parser.add_argument(
        '--min_count', type=int, default=1,
        help="Skip characters that appear fewer than this many times. "
             "Set to 1 to keep everything since our dataset is pretty small. (default: 1)"
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help="Number of training passes over the data. (default: 10)"
    )
    parser.add_argument(
        '--window', type=int, default=5,
        help="How many characters on each side count as 'context'. (default: 5)"
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help="Number of CPU threads to use. More = faster on mltgpu. (default: 4)"
    )
    return parser.parse_args()


def main():
    args = get_args()

    sentences = load_all_sentences(args.tsv_files)

    print(f"\nTraining FastText with dim={args.dim}, window={args.window}, "
          f"min_count={args.min_count}, epochs={args.epochs}...")
    print("(This might take a minute or two depending on the server load.)\n")

    model = FastText(
        sentences=sentences,
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        sg=1,
    )

    model.save(args.output)
    print(f"\nDone! Model saved to: {args.output}")
    print(f"Vocabulary size: {len(model.wv):,} unique tokens")


if __name__ == '__main__':
    main()