from dataclasses import dataclass
import numpy as np


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray

@dataclass
class HW2Dataset:
    train: Split
    val: Split
    test: Split

    @classmethod
    def from_csv(cls, train_path: str, val_path: str, test_path: str):
        train_data = np.loadtxt(train_path, delimiter=",", skiprows=1)
        val_data = np.loadtxt(val_path, delimiter=",", skiprows=1)
        test_data = np.loadtxt(test_path, delimiter=",", skiprows=1)

        train_split = Split(train_data[:, :2], train_data[:, 2])
        val_split = Split(val_data[:, :2], val_data[:, 2])
        test_split = Split(test_data[:, :2], test_data[:, 2])

        return cls(train_split, val_split, test_split)


# Example usage
if __name__ == "__main__":
    ds = HW2Dataset.from_csv("train.csv", "val.csv", "test.csv")
    print(ds.train.X.shape, ds.train.y.shape)
    print(ds.val.X.shape, ds.val.y.shape)
    print(ds.test.X.shape, ds.test.y.shape)