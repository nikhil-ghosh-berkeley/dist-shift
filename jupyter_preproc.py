import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
from scipy.stats import entropy

def probs_to_softacc(probs, ytest):
    """
        probs: array-like of shape (NUM_SAMPLES, NUM_LABELS, NUM_CHECKPOINTS)
        ytest: array-like of shape (NUM_SAMPLES,)
        return softacc: array-like of shape (NUM_SAMPLES, NUM_CHECKPOINTS)
    """
    
    return np.array([probs[i, yi, :] for i, yi in enumerate(ytest)])
    
def smoothen(G, xkey, Ykey, i=None, num_gridpts=50, smooth=True, sigma=2.0, xmin=None, xmax=None):
    """
        Returns a "standard form" for plot of single point at index i
        G : dict-like with data for a given plot group (e.g. Resnet18)
        G[xkey] : array-like of shape (NUM_CHECKPOINTS,)
            the input x-axis (e.g. test accuracies)
        G[Ykey] : array-like of shape (NUM_SAMPLES, NUM_LABELS, NUM_CHECKPOINTS)
            the pointwise hard accuracies
        num_gridpts : int
                      number of gridpoints on x-axis
        sigma : float
               smoothing parameter
    """
    res = dict()
    x = G[xkey]
    Y = G[Ykey] if i is None else G[Ykey][i]
    
    f = interp1d(x, Y, kind='linear')
    
    if xmin is None:
        xmin = np.min(x)
    
    if xmax is None:
        xmax = np.max(x)

        
    x_grid = np.linspace(xmin, xmax, num_gridpts)
    Y_grid = f(x_grid)
    
    res[f'{xkey}_grid'] = x_grid
    res[f'{Ykey}_grid'] = Y_grid
    
    if smooth:
        Y_smooth = gaussian_filter1d(Y_grid, sigma=sigma)
        res[f'{Ykey}_smooth'] = Y_smooth
    
    return res
        
# efficient implementation
def moving_average(x, w):
    """
    x : array-like of shape (..., N)
    w : int
        moving average window-size
    """
    ret = np.cumsum(x, axis=-1)
    ret[..., w:] = ret[..., w:] - ret[..., :-w]
    return ret[..., (w-1):] / w
        
def accuracy(profs, ytest):
    """
    profs : array-like of shape (NUM_SAMPLES, NUM_LABELS, NUM_CHECKPOINTS)
    ytest: array-like of shape (NUM_SAMPLES,)
           true labels of images
    returns acc : array-like of shape (NUM_SAMPLES, NUM_CHECKPOINTS)
    """
    
    return profs[np.arange(profs.shape[0]), ytest]

def get_entropy(profs):
    """
    profs : array-like of shape (NUM_SAMPLES, NUM_LABELS, NUM_CHECKPOINTS)
    ytest: array-like of shape (NUM_SAMPLES,)
           true labels of images
    returns ent : array-like of shape (NUM_SAMPLES, NUM_CHECKPOINTS)
    """
    
    return entropy(profs, axis=1)

def nonmono(prof, WINDOW=4):
    """
    prof : array-like of shape (NUM_SAMPLES, NUM_CHECKPOINTS)
    """
    prof = moving_average(prof, WINDOW)
    d = prof[..., WINDOW:] - prof[..., :-WINDOW] # first derivative
    d[d > 0] = 0
    nm = np.sum(-d, axis=-1) # itegrate only negative part
    return nm

def thresh(prof, WINDOW=4): # score for large jumps
    """
    prof : array-like of shape (NUM_SAMPLES, NUM_CHECKPOINTS)
    """
    WINDOW = 4
    prof = moving_average(prof, WINDOW)
    d = prof[..., WINDOW:] - prof[..., :-WINDOW] # first derivative
    nm = np.max(np.abs(d), axis=-1)
    return nm

def easy(prof):
    return np.mean(prof, axis=-1)

def hard(prof):
    return 1 - easy(prof)

def normalize(a):
    b = a - a.mean(axis=-1)
    b = b / b.std(axis=-1)
    return b