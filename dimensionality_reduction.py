from matplotlib import pyplot as plt
from medmnist import INFO
import numpy as np
from sklearn import decomposition

from utils import COLORS, MedMNISTCategory

def perform_pca(train_feats, test_feats, test_labels, data_flag, legend=True):
    labels_dict = INFO[data_flag]["label"]

    pca = decomposition.PCA(n_components=2)
    pca.fit(train_feats)

    test_feats_reduced = pca.transform(test_feats)

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
