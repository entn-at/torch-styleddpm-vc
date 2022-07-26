from speechset.config import Config as DataConfig
from styleddpm.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, sr: int, hop: int, patch: int):
        """Initializer.
        Args:
            sr: sample rate.
            hop: stft hop length.
            patch: size of the patch.
        """
        # optimizer
        self.learning_rate = 1e-5
        self.beta1 = 0.9
        self.beta2 = 0.999

        # loader settings
        self.batch = 32
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        self.split = -100

        # train iters
        self.epoch = 1000

        # mask ratio
        self.mask_ratio = 0.5

        # segment length
        sec = 1.
        frames = int(sr * sec) // hop
        # quantize with patch size
        self.seglen = frames // patch * patch

        # path config
        self.log = './log'
        self.ckpt = './ckpt'

        # model name
        self.name = 't1'

        # commit hash
        self.hash = 'unknown'


class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.data = DataConfig(batch=None)
        self.model = ModelConfig(self.data.mel)
        self.train = TrainConfig(self.data.sr, self.data.hop, self.model.patch)

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
