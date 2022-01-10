from .base_options import BaseOptions


class ValOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--val_type', type=str, help='volumes2 | cubes ')
        parser.add_argument('--threshold', type=str, default=125, help='probability threshold of the positive class')

        self.isTrain = False
        return parser
