from matplotlib import pyplot as plt
import numpy as np
from sklearn import decomposition

def perform_pca(train_feats, test_feats, test_labels, num_classes):
    pca = decomposition.PCA(n_components=2)
    pca.fit(train_feats)

    test_feats_reduced = pca.transform(test_feats)

    # TODO Refactor
    # Plot the data points coloured by their labels
    plt.figure(figsize=(8, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        plt.scatter(
            test_feats_reduced[test_labels == i, 0],
            test_feats_reduced[test_labels == i, 1],
            color=colors[i],
            label=i,
            alpha=0.2,
        )
    plt.legend()
    plt.show()
