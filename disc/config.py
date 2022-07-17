class Config:
    """Configuration for discriminator.
    """
    def __init__(self, mel: int, speakers: int):
        """Initializer.
        Args:
            mel: size of the spectrogram features.
            speakers: the number of the speakers.
        """
        self.mel = mel
        self.domains = speakers

        # classifier
        self.channels = 128
        self.kernels = 3
        self.stages = 4
        self.blocks = 2
