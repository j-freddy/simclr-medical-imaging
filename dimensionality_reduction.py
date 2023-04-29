from matplotlib import pyplot as plt
from medmnist import INFO
import numpy as np
from sklearn import decomposition
# TODO Try from cuml.manifold import TSNE
# https://medium.com/rapids-ai/tsne-with-gpus-hours-to-seconds-9d9c17c941db
from sklearn.manifold import TSNE

from utils import COLORS, DIMENSIONALITY_REDUCTION_SAMPLES, MedMNISTCategory, get_feats


def perform_feature_analysis(
    train_feats_data,
    test_feats_data,
    test_labels,
    data_flag,
    legend=True,
):
    train_feats = get_feats(train_feats_data)
    test_feats = get_feats(test_feats_data)

    # In SimCLR pretraining we used a batch size of 128 and features = size*4
    assert train_feats.shape[1] == 512
    assert test_feats.shape[1] == 512

    # # Perform PCA
    # test_feats_reduced = perform_pca(train_feats, test_feats)

    # plot_reduced_feats(
    #     test_feats_reduced,
    #     test_labels,
    #     data_flag,
    #     legend,
    # )

    # Perform t-SNE
    test_feats_reduced = perform_tsne(train_feats, test_feats)

    plot_reduced_feats(
        test_feats_reduced,
        test_labels,
        data_flag,
        legend,
    )


def perform_pca(train_feats, test_feats):
    pca = decomposition.PCA(n_components=2)
    pca.fit(train_feats)
    return pca.transform(test_feats)


def perform_tsne(train_feats, test_feats):
    # "It is highly recommended to use another dimensionality reduction method
    # (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the
    # number of dimensions to a reasonable amount (e.g. 50) if the number of
    # features is very high."
    pca = decomposition.PCA(n_components=50)
    pca.fit(train_feats)
    test_feats_pca = pca.transform(test_feats)

    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        # Alternative: https://arxiv.org/pdf/1708.03229.pdf
        perplexity=np.floor(DIMENSIONALITY_REDUCTION_SAMPLES * 0.05),
    )

    return tsne.fit_transform(test_feats_pca)


def plot_reduced_feats(test_feats_reduced, test_labels, data_flag, legend=True):
    labels_dict = INFO[data_flag]["label"]

    # Use stylish plots
    plt.style.use("ggplot")

    num_classes = len(labels_dict)

    # Plot the data points coloured by their labels
    colors = COLORS

    for i in range(num_classes):
        curr_label = labels_dict[str(i)]

        # BloodMNIST: The original 3rd label is very long
        if i == 3 and data_flag == MedMNISTCategory.BLOOD.value:
            curr_label = "immature granulocytes"

        plt.scatter(
            test_feats_reduced[test_labels == i, 0],
            test_feats_reduced[test_labels == i, 1],
            color=colors[i],
            label=curr_label,
            alpha=0.75,
            # Default marker area: 50
            # Make it smaller
            s=16,
        )

        plt.xlabel("principle component 1")
        plt.ylabel("principle component 2")

    if legend:
        plt.legend()
    plt.show()
