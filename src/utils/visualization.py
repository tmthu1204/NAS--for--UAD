import matplotlib.pyplot as plt

def plot_curve(x, y, outpath, title=""):
    # Ensure equal length for x, y
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
