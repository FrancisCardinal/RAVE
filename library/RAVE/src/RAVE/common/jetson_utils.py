import platform


def is_jetson():
    """
    Checks if current platform is likely on a jetson
    """
    return platform.release().split("-")[-1] == "tegra"


def process_video_source(video_source):
    """
    Attempt to convert video_source to int if possible
    """
    try:
        video_source = int(video_source)
    except ValueError:
        pass

    return video_source
