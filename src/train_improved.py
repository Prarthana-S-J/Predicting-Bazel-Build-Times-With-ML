"""Step 3: MODEL B - same model, corrected pairing (features of the change set actually rebuilt)."""
from common import train_and_validate

if __name__ == "__main__":
    train_and_validate("aligned", "improved")
