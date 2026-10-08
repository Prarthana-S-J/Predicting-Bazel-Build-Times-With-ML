"""Step 2: MODEL A - clean paper baseline (paper pairing: features of row i -> CPU time of row i)."""
from common import train_and_validate

if __name__ == "__main__":
    train_and_validate("paper_pairing", "baseline")
