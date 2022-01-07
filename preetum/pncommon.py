import gcsfs
import pickle
import os
import os.path as path
from os.path import join as pjoin
from pathlib import Path

def gopen(gsname, mode='rb'):
    # use token, set by (in order): environmental var, default credential path, default auth
    token = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    tfile = '~/.config/gcloud/application_default_credentials.json'
    if token is None and os.path.isfile(tfile):
        token = tfile

    fs = gcsfs.GCSFileSystem(token=token)
    if gsname.startswith('gs://'):
        gsname = gsname[len('gs://'):]
    return fs.open(gsname, mode)

def gsave(x, gsname):
    with gopen(gsname, 'wb') as f:
        pickle.dump(x, f)
        
def gload(gsname):
    with gopen(gsname, 'rb') as f:
        x = pickle.load(f)
    return x

def glob(gspath):
    fs = gcsfs.GCSFileSystem()
    return fs.glob(gspath)

def dload(gpath, localdir='~/data_cache', crc=True, overwrite=False):
    ''' Downloads object from GCS into localdir (if not exists), and returns the local filename'''
    import subprocess
    localdir = path.expanduser(localdir)
    local_fname = pjoin(localdir, str(abs(hash(gpath))))
    if path.isfile(local_fname) and not overwrite:
        print("file already downloaded:", local_fname)
        return local_fname
    subprocess.call(f'mkdir -p {localdir}', shell=True)
    if not crc:
        # skip CRC hash check (for machines without crcmod installed)
        subprocess.call(f'gsutil -m -o GSUtil:check_hashes=never cp {gpath} {local_fname}', shell=True)
    else:
        subprocess.call(f'gsutil -m cp {gpath} {local_fname}', shell=True)
    return local_fname

