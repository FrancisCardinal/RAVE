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
        self.source = {}
        self.sink = {}

    def add_source(self, source_name, source_type, **kwargs):
        """
        Add source to use for input.
        See PyODAS documentation <https://introlab.github.io/pyodas/_build/html/pyodas/io/io.sources.html>

        Args:
            source_name (str): Name to use to refer to source (dict key)
            source_type (str): Input type to use as source (source_wav, source_mic)
            **kwargs: More keyword args are required depending on input type chosen, .
        Return:
            Source newly added to source dict
        """
        self.source[source_name] = SOURCE_TYPES[source_type.lower()](**kwargs)
        return self.source[source_name]

    def init_sink(self, sink_name, sink_type, **kwargs):
        """
        Initialize sink to use for output.
        See PyODAS documentation <https://introlab.github.io/pyodas/_build/html/pyodas/io/io.sinks.html>

        Args:
            sink_name (str): Name to use to refer to sink (dict key)
            sink_type (str): Output type to use as sink (sink_wav, sink_playback)
            **kwargs: More keyword args are required depending on output type chosen, .
        Return:
            Sink newly added to sink dict
        """
        self.sink[sink_name] = SINK_TYPES[sink_type.lower()](**kwargs)
        return self.sink[sink_name]
