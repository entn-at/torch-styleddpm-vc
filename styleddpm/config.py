class Config:
    """Configuration for StyleDDPM-VC.
    """
    def __init__(self, mel: int):
        """Initializer.
        Args:
            mel: size of the spectrogram features.
        """
        self.mel = mel

        # diffusion steps
        self.steps = 50

        # schedules
        self.internals = 1024
        self.logit_max = 10
        self.logit_min = -10

        # embedder
        self.pe = 128
        self.embeddings = 512
        self.mappings = 2

        # unet
        self.channels = 128
        self.kernels = 3
        self.longrange = 11
        self.stages = 4
        self.blocks = 2

        # style encoder
        self.styles = 32
        self.style_blocks = 2
        self.style_kernels = 3

        # context encoder
        self.patch = 16
        self.pe = 32
        self.heads = 4
        self.ffns = 256
        self.dropout = 0.
        self.layers = 4
        self.dec_kernels = 3
        self.dec_blocks = 2
        self.dec_layers = 2
