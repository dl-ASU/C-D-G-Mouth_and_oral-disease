from .MStrain import train as mstrain
from .SEtrain import train as setrain
from .enstrain import train as enstrain

# Mode Factory that maps modes to classes
TRAIN_FACTORY = {
    "mask": mstrain,
    "site": setrain,
    "none": setrain,
}

def get_train(mode, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if mode not in TRAIN_FACTORY:
        raise ValueError(f"Mode {mode} is not available.")
    return TRAIN_FACTORY[mode](*args, **kwargs)
