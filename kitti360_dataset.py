#%%
import numpy as np
from labels import semantics2rgbimg
import matplotlib.pyplot as plt
from xgutils.vis import visutil
import scipy
from scipy import ndimage
def vis_video(dset_dir="output", 
                    sequence="2013_05_28_drive_0000_sync", 
                    trajs = 400, fps=25):
    img1s, img2s = [], []
    for traj_i in range(trajs):
        fpath = "%s/%s/%08d/labels.npz"%(dset_dir, sequence, traj_i)
        img = plt.imread("%s/%s/%08d/persp_img.png"%(dset_dir, sequence, traj_i))
        img = img[:,:,:-1]
        #img1s.append(img)
        loaded = np.load(fpath)
        img2 = loaded["layout"][0]
        img2 = semantics2rgbimg(img2, vis_color=True)
        #img2s.append(img2)
        #img2 = ndimage.zoom(img2, zoom=(0.5, 0.5, 1.))
        img2 = img2 / 256.
        imgGrid = visutil.imageGrid([img, img2], shape=(1, 2))
        img1s.append(imgGrid)
    #visutil.imgarray2video(targetPath="img1.mp4", img_list=img1s, frameRate=20)
    #visutil.imgarray2video(targetPath="img2.mp4", img_list=img2s, frameRate=20)
    visutil.imgarray2video(targetPath="imgGrid.mp4", img_list=img1s, frameRate=fps)

def export_ditem(dset_dir="output/export/", 
                    sequence="2013_05_28_drive_0000_sync", 
                    traj_i=100):
    os.system("mkdir -p %s/images/%s_%08d" % (dset_dir,sequence, traj_i) )
    os.system("mkdir -p %s/labels/%s_%08d" % (dset_dir,sequence, traj_i) )
    os.system("cp -p %s/images/%s_%08d/0000.png" % (dset_dir,sequence, traj_i) )
    os.system("mkdir -p %s/labels/%s_%08d/0000.png" % (dset_dir,sequence, traj_i) )
def visualize_ditem(dset_dir="output", 
                    sequence="2013_05_28_drive_0000_sync", 
                    traj_i=100):
    img = plt.imread("%s/images/%s_%08d/0000.png" % (dset_dir,sequence, traj_i))
    plt.imshow(img, extent=[-1,1,-1,1])
    plt.show()
    loaded = np.load("%s/labels/%s_%08d/boxes.npz" % (dset_dir,sequence, traj_i))
    for key in loaded:
        print(key, loaded[key].shape)
    img = loaded["layout"][0]
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
    #plt.show()
    print("intrinsic", loaded["intrinsic"])
    return plt.gcf()
if __name__ == "__main__":
    vis_video(trajs=10514, fps=50)
    #visualize_ditem()
    #plt.show()
    #plt.savefig("test.png")
#%%