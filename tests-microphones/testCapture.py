from pyodas.io import MicSource

channels = 2
mic_source = MicSource(channels)

x = mic_source()
print(f'The data is: {x}')