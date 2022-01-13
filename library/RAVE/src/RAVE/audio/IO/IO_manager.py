from pyodas.io import WavSource, MicSource, PlaybackSink, WavSink

SOURCE_TYPES = {
    "source_wav": WavSource,
    "source_mic": MicSource,
}

SINK_TYPES = {
    "sink_wav": PlaybackSink,
    "sink_playback": WavSink,
}


class IOManager:
    """
    Class serving as manager for all input and output operations.
    Can be used for input from microphones or wav file, and for output to device (playback) or wav file.
    See PyODAS documentation <https://introlab.github.io/pyodas/_build/html/pyodas/io/io.html>
    """

    def __init__(self):
        self.source = None
        self.sink = None

    def init_source(self, source_type, **kwargs):
        """
        Initialize source to use for input.
        See PyODAS documentation <https://introlab.github.io/pyodas/_build/html/pyodas/io/io.sources.html>

        Args:
            source_type (str): Input type to use as source (source_wav, source_mic)
            **kwargs: More keyword args are required depending on input type chosen, .
        """
        self.source = SOURCE_TYPES[source_type.lower()](**kwargs)

    def init_sink(self, sink_type, **kwargs):
        """
        Initialize sink to use for output.
        See PyODAS documentation <https://introlab.github.io/pyodas/_build/html/pyodas/io/io.sinks.html>

        Args:
            sink_type (str): Output type to use as sink (sink_wav, sink_playback)
            **kwargs: More keyword args are required depending on output type chosen, .
        """
        self.sink = SINK_TYPES[sink_type.lower()](**kwargs)