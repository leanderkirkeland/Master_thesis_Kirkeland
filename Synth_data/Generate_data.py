from scipy.__config__ import show
from git.propagating_blobs.blobmodel import Model, show_model, BlobFactory, Blob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


for i in range(100):
    bm = Model(
        Nx=32,
        Ny=64,
        Lx=32,
        Ly=64,
        dt=1,
        T=64,
        periodic_y=False,
        blob_shape="gauss",
        num_blobs= np.random.randint(20)+1,
        t_drain=1e10,
        labels="inorder",
    )

    ds = bm.make_realization(speed_up=True, error=1e-2, file_name=f"Tracking_{i}.nc")

show_model(ds, save = False, gif_name="Realim.gif", variable="blob_labels")

np.random.normal(6,2, size= 20).min()