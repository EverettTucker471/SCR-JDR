import re
import argparse
import matplotlib.pyplot as plt

def parse_log(filename):
    global_epoch = 0

    epochs = []
    train_acc = []

    eval_epochs = []
    val_acc = []
    test_acc = []

    with open(filename, "r") as f:
        for line in f:

            # ---------------------------------
            # TRAINING LINE
            # ---------------------------------
            m_train = re.search(
                r"Epoch\s+(\d+).*Train acc:\s*([0-9.]+)", line
            )

            if m_train:
                # IGNORE the local epoch number (resets each stage)
                acc = float(m_train.group(2))

                epochs.append(global_epoch)
                train_acc.append(acc)

                global_epoch += 1
                continue

            # ---------------------------------
            # EVALUATION LINE
            # ---------------------------------
            m_eval = re.search(r"Val\s+([0-9.]+).*Test\s+([0-9.]+)", line)

            if m_eval:
                val = float(m_eval.group(1)) * 100
                test = float(m_eval.group(2)) * 100

                # evaluation occurs *after* the latest epoch
                eval_epochs.append(global_epoch - 1)
                val_acc.append(val)
                test_acc.append(test)

                continue

    return epochs, train_acc, eval_epochs, val_acc, test_acc


def plot_graph(epochs, train_acc, eval_epochs, val_acc, test_acc, filename):
    plt.figure(figsize=(12,6))

    plt.plot(epochs, train_acc, label="Train Accuracy (%)", linewidth=1.5)
    plt.plot(eval_epochs, val_acc, "o-", label="Validation Accuracy (%)", markersize=4)
    plt.plot(eval_epochs, test_acc, "s-", label="Test Accuracy (%)", markersize=4)

    plt.xlabel("Global Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training/Validation/Test Accuracy (Global Epoch Index)\n{filename}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, required=True)
    args = parser.parse_args()

    epochs, train_acc, eval_epochs, val_acc, test_acc = parse_log(args.logfile)
    plot_graph(epochs, train_acc, eval_epochs, val_acc, test_acc, args.logfile)
