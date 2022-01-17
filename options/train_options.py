from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate for adam')
        parser.add_argument('--n_epochs', type=int, default=100, help='epochs of train, test, evaluating')
        parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model')
        parser.add_argument('--batch_size', type=int, default=30, help='batch size')
        parser.add_argument('--artifacts', type=bool, default=True, help='train with artifacts volumes')
        parser.add_argument('--n_samples', type=int, default=4, help='crop n volumes from one cube')
        parser.add_argument('--semi', type=bool, default=False, help='train with semi-supervised methods')
        parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle the data')
        parser.add_argument('--shuffle_val', type=bool, default=True, help='whether to shuffle the val data')
        parser.add_argument('--exp', type=str, default='Sunmap', help='name of the experiment')
        parser.add_argument('--checkpoint_name', type=str, help='name of the experiment')
        parser.add_argument('--net', type=str, help='unet | axialunet')

        self.isTrain = True
        return parser
