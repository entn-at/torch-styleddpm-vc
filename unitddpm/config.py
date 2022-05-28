class Config:
    """Configuration for UNIT-DDPM.
    """
    def __init__(self, mel: int):
        """Initializer.
        Args:
            mel: size of the spectrogram features.
        """
        self.mel = mel

        # embedder
        self.pe = 128
        self.embeddings = 512
        self.mappings = 2

        # unet
        self.channels = 128
        self.kernels = 3
        self.stages = 4
        self.blocks = 2
