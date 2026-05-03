import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics

sns.set(style="whitegrid")


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_al_learning_curve(metric_df, out_path, title="Learning Curve"):
    """metric_df: pandas.DataFrame with index=rounds (or labelled size) and columns=metrics"""
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6,4))
    for col in metric_df.columns:
        ax.plot(metric_df.index, metric_df[col], marker='o', label=col)
    ax.set_xlabel('Round')
    ax.set_ylabel('Metric')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_al_acquisition_distributions(acq_scores_by_round, out_path, title="Acquisition distributions"):
    _ensure_dir(out_path)
    import pandas as pd
    # acq_scores_by_round: list of arrays
    rows = []
    for r, arr in enumerate(acq_scores_by_round):
        for v in np.asarray(arr).ravel():
            rows.append({'round': r, 'score': float(v)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x='round', y='score', data=df, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_al_selection_timeline(selection_matrix, out_path, title="Selection timeline"):
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(selection_matrix.T.astype(int), cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Pool index')
    ax.set_ylabel('Round')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_al_uncertainty_vs_error(df_by_round, measure, out_path, title=None):
    _ensure_dir(out_path)
    # df_by_round: list of DataFrames or tuples (scores_df, errors)
    import pandas as pd
    rows = []
    for r, (scores_df, errors) in enumerate(df_by_round):
        sc = scores_df[measure].values
        for i, v in enumerate(sc):
            rows.append({'round': r, 'score': float(v), 'error': float(errors[i])})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x='score', y='error', hue='round', data=df, palette='viridis', ax=ax, legend='full')
    ax.set_title(title or f'{measure}: uncertainty vs error')
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_ood_score_distributions(df_id, df_ood, measures, out_path, title='ID vs OOD score distributions'):
    _ensure_dir(out_path)
    import pandas as pd
    M = len(measures)
    cols = measures
    fig, axes = plt.subplots(max(1, M), 1, figsize=(6, 2*M))
    if M == 1:
        axes = [axes]
    for ax, m in zip(axes, cols):
        idv = df_id[m].values
        oov = df_ood[m].values
        sns.kdeplot(idv, label='ID', ax=ax)
        sns.kdeplot(oov, label='OOD', ax=ax)
        ax.set_title(m)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_ood_roc_pr(df_id, df_ood, measures, out_path, title='ROC and PR curves'):
    _ensure_dir(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    for m in measures:
        y_id = np.zeros(len(df_id))
        y_ood = np.ones(len(df_ood))
        scores = np.concatenate([df_id[m].values, df_ood[m].values])
        labels = np.concatenate([y_id, y_ood])
        try:
            fpr, tpr, _ = metrics.roc_curve(labels, scores)
            roc_auc = metrics.auc(fpr, tpr)
            axes[0].plot(fpr, tpr, label=f'{m} (AUC={roc_auc:.2f})')
        except Exception:
            pass
        try:
            precision, recall, _ = metrics.precision_recall_curve(labels, scores)
            ap = metrics.average_precision_score(labels, scores)
            axes[1].plot(recall, precision, label=f'{m} (AP={ap:.2f})')
        except Exception:
            pass

    axes[0].set_title('ROC curves')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].legend()
    axes[1].set_title('Precision-Recall')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_ood_summary_bars(metrics_summary, out_path, title='OOD summary'):
    _ensure_dir(out_path)
    import pandas as pd
    # metrics_summary: dict measure -> {'auroc': , 'aupr': }
    rows = []
    for m, v in metrics_summary.items():
        rows.append({'measure': m, 'auroc': v.get('auroc'), 'aupr': v.get('aupr')})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['auroc'], width, label='AUROC')
    ax.bar(x + width/2, df['aupr'], width, label='AUPR')
    ax.set_xticks(x)
    ax.set_xticklabels(df['measure'], rotation=45, ha='right')
    ax.set_ylim(0,1)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    return fig
