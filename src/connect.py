import os
from pickle import TRUE
import mne
import numpy as np
from typing import Sequence, List, Tuple, Union, Optional, Dict, Any
from collections.abc import Callable
from copy import deepcopy
from mne_connectivity import seed_target_indices, spectral_connectivity_epochs,spectral_connectivity_time
from mne.viz import plot_topomap, plot_alignment
from mayavi import mlab
import matplotlib
from matplotlib import pyplot as plt, colors, transforms
import numpy as np
from os.path import join, exists
#import torch
# matplotlib.use('Agg') # saves ram https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close
# mlab.init_notebook("x3d", 800, 600)
# mne.viz.set_3d_backend('notebook')
from mne.io.constants import FIFF
from mne.io.pick import _picks_to_idx
from mne.utils import _validate_type, fill_doc, verbose
import ntpath


def plot_sensors_connectivity(info, con, picks=None,
                              cbar_label='Connectivity',NMAX=100):
    """Visualize the sensor connectivity in 3D.

    Parameters
    ----------
    info : dict | None
        The measurement info.
    con : array, shape (n_channels, n_channels) | Connectivity
        The computed connectivity measure(s).
    %(picks_good_data)s
        Indices of selected channels.
    cbar_label : str
        Label for the colorbar.

    Returns
    -------
    fig : instance of Renderer
        The 3D figure.
    """
    _validate_type(info, "info")

    from mne.viz.backends.renderer import _get_renderer
    from mne_connectivity.base import BaseConnectivity

    if isinstance(con, BaseConnectivity):
        con = con.get_data()

    renderer = _get_renderer(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

    picks = _picks_to_idx(info, picks)
    if len(picks) != len(con):
        raise ValueError('The number of channels picked (%s) does not '
                         'correspond to the size of the connectivity data '
                         '(%s)' % (len(picks), len(con)))

    # Plot the sensor locations
    sens_loc = [info['chs'][k]['loc'][:3] for k in picks]
    sens_loc = np.array(sens_loc)

    renderer.sphere(np.c_[sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2]],
                    color=(1, 1, 1), opacity=1, scale=0.005)

    # Get the strongest connections
    n_con = NMAX  # show up to 20 connections
    min_dist = 0.05  # exclude sensors that are less than 5cm apart
    threshold = np.sort(con, axis=None)[-n_con]
    ii, jj = np.where(con >= threshold)

    # Remove close connections
    con_nodes = list()
    con_val = list()
    for i, j in zip(ii, jj):
        if np.linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
            con_nodes.append((i, j))
            con_val.append(con[i, j])

    con_val = np.array(con_val)

    # Show the connections as tubes between sensors
    vmax = np.max(con_val)
    vmin = np.min(con_val)
    for val, nodes in zip(con_val, con_nodes):
        x1, y1, z1 = sens_loc[nodes[0]]
        x2, y2, z2 = sens_loc[nodes[1]]
        tube = renderer.tube(origin=np.c_[x1, y1, z1],
                             destination=np.c_[x2, y2, z2],
                             scalars=np.c_[val, val],
                             vmin=vmin, vmax=vmax,
                             reverse_lut=True)

    renderer.scalarbar(source=tube, title=cbar_label)

    # Add the sensor names for the connections shown
    nodes_shown = list(set([n[0] for n in con_nodes] +
                           [n[1] for n in con_nodes]))

    for node in nodes_shown:
        x, y, z = sens_loc[node]
        renderer.text3d(x, y, z, text=info['ch_names'][picks[node]],
                        scale=0.005,
                        color=(0, 0, 0))

    renderer.set_camera(azimuth=-88.7, elevation=40.8,
                        distance=0.76,
                        focalpoint=np.array([-3.9e-4, -8.5e-3, -1e-2]))
    renderer.show()
    return renderer.scene()


def standardize(name):
    std_name = name.strip(".")
    std_name = std_name.upper()
    if std_name.endswith("Z"):
        std_name = std_name[:-1] + "z"
    if std_name.startswith("FP"):
        std_name = "Fp" + std_name[2:]
    return std_name

def eegbci_connectivity(
    datapath: str = "data",
    epoch_duration: Union[float, int] = 5,
    band: Tuple[float, float] = (0., 45.0),
    outpath: str = "data/connectivity",
):

    # Make sure path exist
    if datapath is not None:
        # Use default MNE dir
        os.makedirs(datapath, exist_ok=True)
    os.makedirs(outpath, exist_ok=True)

    # Download the dataset
    filenames = {}
    sub_list = list(range(1, 110))
    task_list = (0, 1)  # ('close','open')
    task_map={0:'close',1:'open'}
    for sub in sub_list:
        filenames[sub] = {}
        filenames[sub][task_list[0]] = mne.datasets.eegbci.load_data(
            sub, [2], path=datapath, update_path=False, verbose=False
        )[
            0
        ]  # 2 Baseline, eyes closed
        filenames[sub][task_list[1]] = mne.datasets.eegbci.load_data(
            sub, [1], path=datapath, update_path=False, verbose=False
        )[
            0
        ]  # 1 Baseline, eyes open

    chan_names = []
    methods = ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased']
    # delta (0-4 Hz), theta (4-8 Hz), alpha (8-12 Hz) and beta (12-30 Hz), as well as a combined 0-30 Hz band
    bands_fmin = (2.5 , 4 ,  8 , 12 , 2.5)
    bands_fmax = (4   , 8 , 12 , 30 ,  30)
    bands_label =['delta','theta','alpha','beta','wide']
    #RuntimeWarning: fmin=0.000 Hz corresponds 
    #to 0.000 < 5 cycles based on the epoch length 2.000 sec, need at least inf sec epochs or fmin=2.500. Spectrum estimate will be unreliable.
    bands = {label:(x,y) for label,x,y in zip(bands_label,bands_fmin,bands_fmax)}
    montage_type = "standard_1005"
    # define channel names

    for method in methods:
        for sub in sub_list:
            for task in task_list:
                outfile = ntpath.basename(filenames[sub][task].replace('.edf',f'_{task_map[task]}_{method}.npy'))
                if os.path.isfile(outfile):
                    print(f'{outfile} already existed... skipping...')
                else:
                    raw = mne.io.read_raw_edf(
                        filenames[sub][task], preload=True, verbose=False
                    )
                    raw = raw.copy().filter(l_freq=band[0] , h_freq=band[1])
                    raw = raw.rename_channels(standardize)
                    raw = raw.drop_channels(["T9","T10","Iz"])
                    raw = raw.set_montage("standard_1005")
                    # Segment into epochs
                    epochs = mne.make_fixed_length_epochs(
                        raw, duration=epoch_duration, preload=True, verbose=False
                    )
                    del raw
                    chan_names.append(deepcopy(epochs.info["ch_names"]))
                    #epochs.plot_sensors(kind='3d',block=True)
                    conn = spectral_connectivity_epochs(
                        epochs.get_data(),mode='multitaper', indices=None,
                        sfreq=epochs.info['sfreq'], fmin=bands_fmin,fmax=bands_fmax,n_jobs = 4,method=method,faverage=True,names=epochs.info['ch_names'])
                    # x_idx,y_idx = topk(conn.get_data('dense')[:,:,b],100)
                    # conn_mask=conn.get_data('dense')[:,:,b]
                    # conn_mask[x_idx,y_idx]=np.inf
                    # other_idx = np.where(conn_mask!=np.inf)
                    # conn_mask= conn.get_data('dense')[:,:,b]
                    # conn_mask[other_idx]=0

                    #freqs = coh.freqs
                    #cohs.append(coh.get_data())
                    # combined_view(epochs.info, conn)
                    results = {}
                    results['values']=conn.get_data('dense')
                    results['dims']=('node_in','node_out',conn.dims[-1])
                    results['attrs']=conn.attrs
                    results['bands']=bands
                    results['method']=method
                    results['freqs']=conn.freqs
                    np.save(outfile,results)
                    #plot_sensors_connectivity(epochs.info,conn.get_data('dense')[:,:,b],cbar_label=method+'-'+task,NMAX=100)
                    
                    #print('ok')

if __name__=='__main__':
    eegbci_connectivity(None,2)