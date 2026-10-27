from .onyx import open_onyx
from .decimated import open_decimated

def open_das(path, instrument, **kwargs):
    instrument = instrument.lower()

    if instrument == "onyx":
        return open_onyx(path, **kwargs)
    if instrument == "decimated":
        return open_decimated(path, **kwargs)

    raise ValueError(f"Unknown instrument: {instrument}")