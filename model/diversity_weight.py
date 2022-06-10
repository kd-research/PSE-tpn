import numpy as np
import shelve
from collections import deque
from torch import Tensor

latent_length = 256 * 8


def diversity_group(current, samples):
    if isinstance(current, Tensor):
        current = current.detach().cpu().numpy()
    current = current.flatten()
    distance = np.power(current - samples, 2).mean(1)
    group = np.argmin(distance)
    print(group, ends="\t")
    return group


def diversity_weight(current):
    """
    Compute diversity weight of current latent vector from history latent vectors
    Args:
        current: list[float]

    Returns: float
    """

    with shelve.open('results/.variables') as db:
        if 'samples' not in db:
            db['samples'] = np.random.randn(100, latent_length)
        thisIdx = diversity_group(current, db['samples'])
        history = db.get('history', deque(maxlen=1000))
        history.append(thisIdx)
        db['history'] = history
        occurence = sum(map(lambda x: x == thisIdx, history))
        print(len(history))
    return 1.0 / occurence
