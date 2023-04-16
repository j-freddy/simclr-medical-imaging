import argparse


# TODO Get rid of the methods in utils.py and use these instead.

class Arguments:
    @staticmethod
    def parse_train(downstream=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", type=str, help="Data category", required=True)
        parser.add_argument("-epochs", type=int,
                            help="Maximum number of epochs", required=True)
        # Optional. Default is to use all samples
        parser.add_argument("-samples", type=int, help="Number of samples")

        if not downstream:
            # Optional. Default is new ResNet model.
            parser.add_argument(
                "-fin", type=str, help="Initial model (to further pretrain)")

        if downstream:
            # Optional. Default is new ResNet model.
            parser.add_argument(
                "-fin",
                type=str,
                help="Pretrained model filename"
            )

        # Optional. Default is "[pretrain/downstream]-[category]"
        parser.add_argument("-fout", type=str, help="Output model filename")

        args = parser.parse_args()

        if args.fin:
            args.fin += ".ckpt"
        return args.c, args.epochs, args.samples, args.fin, args.fout

    @staticmethod
    def parse_test(logistic_regression=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", type=str, help="Data category", required=True)

        if logistic_regression:
            # Logistic regression model requires a separate base encoder
            parser.add_argument(
                "-fencoder",
                type=str,
                help="Base encoder filename",
                required=True
            )

        parser.add_argument(
            "-fin",
            type=str,
            help="Model filename",
            required=True
        )

        args = parser.parse_args()
        args.fin += ".ckpt"

        if logistic_regression:
            return args.c, args.fencoder, args.fin
        return args.c, args.fin

    @staticmethod
    def parse_pca(logistic_regression=False):
        return Arguments.parse_test(logistic_regression)
