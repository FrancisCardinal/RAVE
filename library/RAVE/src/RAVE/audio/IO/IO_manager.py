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

    def __init__(self):
        self.source = None
        self.sink = None

    def init_source(self, source_type, **kwargs):
        self.source = SOURCE_TYPES[source_type.lower()](**kwargs)

    def init_sink(self, sink_type, **kwargs):
        self.sink = SINK_TYPES[sink_type.lower()](**kwargs)