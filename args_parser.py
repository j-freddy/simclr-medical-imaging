import argparse


class Arguments:
    """
    Utility class containing methods for argument parsing across different
    Python scripts.

    This class should never be instantiated.
    """

    @staticmethod
    def parse_args_train(downstream=False):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", type=str, help="Data category", required=True)
        parser.add_argument("-epochs", type=int,
                            help="Maximum number of epochs", required=True)

        if not downstream:
            parser.add_argument(
                "-aug",
                type=str,
                help="Augmentation sequence: natural/novel",
                required=True,
            )

        # Optional - Default is to use all samples
        parser.add_argument("-samples", type=int,
                            help="Number of samples", default=-1)

        if not downstream:
            # Optional - Default is new ResNet model
            parser.add_argument(
                "-fin", type=str, help="Initial model (to further pretrain)"
            )

        if downstream:
            # Optional - Default is to use entire dataset
            parser.add_argument(
                "-spc", type=int, help="Number of samples per class", default=-1)
            # Optional - Default is new ResNet model.
            parser.add_argument(
                "-fin", type=str, help="Pretrained model filename")

        # Optional - Default is "[pretrain/downstream]-[category]"
        parser.add_argument("-fout", type=str, help="Output model filename")

        args = parser.parse_args()

        if args.fin:
            args.fin += ".ckpt"

        if downstream:
            return args.c, args.epochs, args.spc, args.samples, args.fin, args.fout
        return args.c, args.epochs, args.aug, args.samples, args.fin, args.fout

    @staticmethod
    def parse_args_test(logistic_regression=False):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", type=str, help="Data category", required=True)

        if logistic_regression:
            # Logistic regression model requires a separate base encoder
            parser.add_argument("-fencoder", type=str,
                                help="Base encoder filename", required=True)

        parser.add_argument(
            "-fin", type=str, help="Model filename", required=True)

        args = parser.parse_args()
        args.fin += ".ckpt"

        if logistic_regression:
            args.fencoder += ".ckpt"
            return args.c, args.fencoder, args.fin
        return args.c, args.fin

    @staticmethod
    def parse_args_feature_analysis():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", type=str, help="Data category", required=True)
        parser.add_argument(
            "-fin", type=str, help="Model filename", required=True)
        parser.add_argument(
            "-tsne",
            action=argparse.BooleanOptionalAction,
            help="Explore t-SNE in-depth using various perplexities",
            default=False,
        )
        parser.add_argument(
            "-legend",
            action=argparse.BooleanOptionalAction,
            help="Show legend in plots",
            default=False,
        )

        args = parser.parse_args()
        args.fin += ".ckpt"

        return args.c, args.fin, args.tsne, args.legend

    @staticmethod
    def parse_args_img_viewer():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", type=str, help="Data category", required=True)
        parser.add_argument(
            "-aug",
            type=str,
            help="Augmentation sequence: natural/novel",
            required=True,
        )

        args = parser.parse_args()
        return args.c, args.aug

    @staticmethod
    def parse_data_flag():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", type=str, help="Data category", required=True)
        
        args = parser.parse_args()

        return args.c
