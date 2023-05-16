import os
import sys
from matplotlib import pyplot as plt
from medmnist import INFO
import numpy as np
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch

from utils import COLORS, DIMENSIONALITY_REDUCTION_SAMPLES, OUT_PATH, MedMNISTCategory, get_feats


def perform_feature_analysis(
    train_feats_data,
    test_feats_data,
    train_labels,
    test_labels,
    data_flag,
    legend=True,
    explore_tsne_only=True,
):
    train_feats = get_feats(train_feats_data)
    test_feats = get_feats(test_feats_data)

    # In SimCLR pretraining we used a batch size of 128 and features = size*4
    assert train_feats.shape[1] == 512
    assert test_feats.shape[1] == 512

    os.makedirs(OUT_PATH, exist_ok=True)

    if not explore_tsne_only:
        # Perform PCA
        test_feats_reduced = perform_pca(train_feats, test_feats)

        plot_reduced_feats(
            test_feats_reduced,
            test_labels,
            data_flag,
            legend,
            # filter_indices=[5],
        )

        fig_name = f"pca-{data_flag}.png"
        plt.savefig(os.path.join(OUT_PATH, fig_name))

        print(f"Silhouette score: {silhouette_score(test_feats_reduced, test_labels)}")

        # Perform t-SNE
        test_feats_reduced = perform_tsne(
            train_feats,
            test_feats,
            perplexity=30,
        )

        plot_reduced_feats(
            test_feats_reduced,
            test_labels,
            data_flag,
            legend,
            # filter_indices=[5],
            component_label="component",
        )

        fig_name = f"tsne-{data_flag}.png"
        plt.savefig(os.path.join(OUT_PATH, fig_name))
    else:
        # We use this environment if t-SNE with default complexity does not
        # produce clusters

        n, _ = train_feats.shape

        # Try perplexity [5, 10, 15, ..., 100]
        perplexities = np.arange(5, 101, 5)

        for perplexity in perplexities:
            print(f"Current perplexity: {perplexity}")

            # Perform t-SNE on train data
            indices = torch.randperm(n)[:DIMENSIONALITY_REDUCTION_SAMPLES]
            pseudo_test_feats = train_feats[indices]
            pseudo_test_labels = train_labels[indices]

            train_feats_reduced = perform_tsne(
                train_feats,
                pseudo_test_feats,
                perplexity=perplexity,
            )

            plot_reduced_feats(
                train_feats_reduced,
                pseudo_test_labels,
                data_flag,
                legend=False,
                component_label="component",
            )

            fig_name = f"tsne-{data_flag}-{perplexity}-train.png"
            plt.savefig(os.path.join(OUT_PATH, fig_name))

            # Uncomment this to filter some labels
            plot_reduced_feats(
                train_feats_reduced,
                pseudo_test_labels,
                data_flag,
                legend=False,
                filter_indices=[5],
                component_label="component",
            )

            fig_name = f"tsne-{data_flag}-{perplexity}-train-filtered.png"
            plt.savefig(os.path.join(OUT_PATH, fig_name))

            # Perform t-SNE on test data
            test_feats_reduced = perform_tsne(
                train_feats,
                test_feats,
                perplexity=perplexity,
            )

            plot_reduced_feats(
                test_feats_reduced,
                test_labels,
                data_flag,
                legend=False,
                component_label="component",
            )

            fig_name = f"tsne-{data_flag}-{perplexity}-test.png"
            plt.savefig(os.path.join(OUT_PATH, fig_name))

            # Uncomment this to filter some labels
            plot_reduced_feats(
                test_feats_reduced,
                test_labels,
                data_flag,
                legend=False,
                filter_indices=[5],
                component_label="component",
            )

            fig_name = f"tsne-{data_flag}-{perplexity}-test-filtered.png"
            plt.savefig(os.path.join(OUT_PATH, fig_name))


def perform_pca(train_feats, test_feats):
    pca = decomposition.PCA(n_components=2)
    pca.fit(train_feats)

    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Sum: {pca.explained_variance_ratio_.sum()}")

    return pca.transform(test_feats)


def perform_tsne(train_feats, test_feats, perplexity):
    # "It is highly recommended to use another dimensionality reduction method
    # (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the
    # number of dimensions to a reasonable amount (e.g. 50) if the number of
    # features is very high."
    # 
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    pca = decomposition.PCA(n_components=50)
    pca.fit(train_feats)

    feats_pca = pca.transform(test_feats)

    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        # Alternative: https://arxiv.org/pdf/1708.03229.pdf
        perplexity=perplexity,
    )

    return tsne.fit_transform(feats_pca)


def plot_reduced_feats(feats_reduced, labels, data_flag, legend=True, filter_indices=[], component_label="principle component"):
    labels_dict = INFO[data_flag]["label"]

    # Use stylish plots
    plt.style.use("ggplot")

    # Clear plot
    plt.clf()

    num_classes = len(labels_dict)

    # Plot the data points coloured by their labels
    colors = COLORS

    for i in range(num_classes):
        if i in filter_indices:
            continue

        curr_label = labels_dict[str(i)]

        # BloodMNIST: The original 3rd label is very long
        if i == 3 and data_flag == MedMNISTCategory.BLOOD.value:
            curr_label = "immature granulocytes"

        plt.scatter(
            feats_reduced[labels == i, 0],
            feats_reduced[labels == i, 1],
            color=colors[i],
            label=curr_label,
            alpha=0.75,
            # Default marker area: 50
            # Make it smaller
            s=16,
        )

        plt.xlabel(f"{component_label} 1")
        plt.ylabel(f"{component_label} 2")

    if legend:
        plt.legend()
