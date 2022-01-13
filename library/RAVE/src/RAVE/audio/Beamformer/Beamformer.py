from pyodas.core import DelaySum, Mvdr, Gev

BEAMFORMER_TYPES = {
    "delaysum": DelaySum,
    "mvdr": Mvdr,
    "gev": Gev
}


class Beamformer:

    def __init__(self, name, **kwargs):
        self.beamformer_type = name.lower()
        self.beamformer = BEAMFORMER_TYPES[name.lower()](**kwargs)

    def __call__(self, **kwargs):
        return self.beamformer(**kwargs)
