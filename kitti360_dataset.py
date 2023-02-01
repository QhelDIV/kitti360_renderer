import numpy as np
from labels import semantics2rgbimg
import matplotlib.pyplot as plt

def visualize_ditem(dset_dir="output", 
                    sequence="2013_05_28_drive_0000_sync", 
                    traj_i=0):
    fpath = "%s/%s/%08d/labels.npz"%(dset_dir, sequence, traj_i)
    #img = plt.imread("output/2013_05_28_drive_0000_sync/00000100/persp_img.png")

    loaded = np.load(fpath)
    img = loaded["layout"]
    img = semantics2rgbimg(img, vis_color=True)
    plt.imshow(img, extent=[-1,1,-1,1])
    xx, xt= loaded["camera_coords"], loaded["target_coords"]
    plt.scatter(xx[:,0], xx[:,2], zorder=100, color='r', s=0.5)
    for i in range(len(xx)):
        plt.plot(   [xx[i,0],  xt[i,0]],
                    [xx[i,2],  xt[i,2]])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.show()