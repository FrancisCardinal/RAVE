from pyodas.core import DelaySum, Mvdr, Gev

BEAMFORMER_TYPES = {
    "delaysum": DelaySum,
    "mvdr": Mvdr,
    "gev": Gev
}


class Beamformer:
    """
    Class that handles Beamformer type abstraction and all external calls
    See PyOdas beamformer documentation <https://introlab.github.io/pyodas/_build/html/pyodas/core/beamformers.html>.

    Args:
        name (str): Beamformer type name (delaysum, mvdr, gev)
        **kwargs: Keyword arguments relative to the initialization of the various beamformers
    """

    def __init__(self, name, **kwargs):
        self.beamformer_type = name.lower()
        self.beamformer = BEAMFORMER_TYPES[name.lower()](**kwargs)

    def __call__(self, **kwargs):
        """
        Execute beamformer implementation (execution depends on beamformer chosen)
        See PyOdas beamformer documentation
        <https://introlab.github.io/pyodas/_build/html/pyodas/core/beamformers.html>.
        """
        args = list(kwargs.values())
        return self.beamformer(*args)
