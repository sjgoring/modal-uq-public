
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap(df, title=None, save_path=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis')
    if title: plt.title(title)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    return plt.gcf()
