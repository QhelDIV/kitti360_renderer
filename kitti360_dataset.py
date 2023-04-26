#%%
import numpy as np
from labels import semantics2rgbimg, kitti360_sequences
import matplotlib.pyplot as plt
from xgutils.vis import visutil
from xgutils import sysutil
import scipy
from scipy import ndimage
def add_traj(img, xx, xt, resolution=(256,256)):
    fig, ax = visutil.newPlot(resolution=resolution)
    ax.imshow(img, extent=[-1,1,-1,1])
    ax.scatter(xx[:,0], xx[:,1], zorder=100, color='r', s=3.5)
    cmap = plt.get_cmap('viridis')
    for i in range(len(xx)):
        ax.plot(   [xx[i,0],  xt[i,0]],
                    [xx[i,1],  xt[i,1]], color=cmap(i/len(xx)))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.axis('off')
    img = visutil.fig2img(fig)
    return img
def render_imgs(dset_dir="output",
                    sequence="2013_05_28_drive_0000_sync", 
                    trajlist = range(400)):
    imgs = []
    for traj_i in sysutil.progbar(trajlist):
        fpath = "%s/labels/%s_%08d/boxes.npz"%(dset_dir, sequence, traj_i)
        save_path = "%s/datavis/imgs/%s/%08d.png"%(dset_dir, sequence, traj_i)
        if os.path.exists(save_path):
            continue
        img = plt.imread("%s/images/%s_%08d/0000.png"%(dset_dir, sequence, traj_i))
        img = img[:,:,:-1]

        loaded = np.load(fpath)
        img2 = loaded["layout"][0]
        img2 = semantics2rgbimg(img2, vis_color=True)
        img2 = add_traj(img2, loaded["camera_coords"], loaded["target_coords"], resolution = (img2.shape[0], img2.shape[1]))

        img2 = img2[:,:,:-1]
        img3 = loaded["layout_noveg"][0]
        img3 = semantics2rgbimg(img3, vis_color=True)/256.

        imgGrid = visutil.imageGrid([img, img2, img3], shape=(1, -1))
        #imgs.append(imgGrid) # The memory cost is too high
        sysutil.mkdir("%s/datavis/imgs/%s"%(dset_dir, sequence))
        visutil.saveImg(save_path, imgGrid)
    return None
import os
def export_datavis_videos(dset_dir="output", filters={},
                    sequence="2013_05_28_drive_0000_sync", 
                    total_trajs = (400), fps=60):
    imgs = render_imgs(dset_dir, sequence, list(range(total_trajs)))
    imgs = np.array(imgs)
    print("Exporting videos of sequence %s" % sequence)
    for filter_name in filters:
        outdir = "%s/datavis/" % (dset_dir)
        out_path="%s/%s_%s.mp4"%(outdir, filter_name, sequence)
        imgs_dir = "%s/datavis/imgs/%s"%(dset_dir, sequence)
        fimgs_dir = "%s_%s/"%(imgs_dir, filter_name)
        sysutil.mkdir(imgs_dir)
        sysutil.mkdir(fimgs_dir)
        filt = filters[filter_name]
        for i,fi in enumerate(np.where(filt)[0]):
            os.system("cp %s/%08d.png %s/%08d.png"%(imgs_dir, fi, fimgs_dir, i))

        visutil.imgs2video2(imgs_dir = fimgs_dir, out_path=out_path, frameRate=fps, ind_pattern="%08d.png")
        os.system("rm -r %s"%(fimgs_dir))

def export_ditem(   output_dir="output/export/", 
                    dset_dir="output",
                    sequence="2013_05_28_drive_0000_sync", 
                    traj_i=100):
    os.system("mkdir -p %s/images/%s_%08d" % (output_dir,sequence, traj_i) )
    os.system("mkdir -p %s/labels/%s_%08d" % (output_dir,sequence, traj_i) )

    ipath = "%s/images/%s_%08d/0000.png"%(dset_dir, sequence, traj_i)
    opath = "%s/images/%s_%08d/0000.png"%(output_dir, sequence, traj_i)
    print(ipath, opath)
    os.system("cp %s %s" % (ipath, opath) )

    ipath = "%s/labels/%s_%08d/boxes.npz"%(dset_dir, sequence, traj_i)
    opath = "%s/labels/%s_%08d/boxes.npz"%(output_dir, sequence, traj_i)
    print(ipath, opath)
    os.system("cp %s %s" % (ipath, opath) )

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
    xx, xt= loaded["camera_coords"], loaded["target_coords"]
    img = add_traj(img, xx, xt)
    plt.imshow(img)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    #plt.show()
    print("intrinsic", loaded["intrinsic"])
    print("denormed_intrinsic", loaded["denormed_intrinsic"])
    return plt.gcf()

def test_ditem(sp, dset_dir="output", 
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
    xx, xt = sp.export_next_cams(traj_i)
    plt.scatter(xx[:,0], xx[:,1], zorder=100, color='r', s=0.5)
    for i in range(len(xx)):
        plt.plot(   [xx[i,0],  xt[i,0]],
                    [xx[i,1],  xt[i,1]])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    #plt.show()
    print("intrinsic", loaded["intrinsic"])
    return plt.gcf()


def test_load():
    loaded = np.load("output/labels/2013_05_28_drive_0000_sync_00000100/boxes.npz")
    for key in loaded:
        print(key, loaded[key].shape)
        print(loaded[key])
    img = loaded["layout"][0]
    img = semantics2rgbimg(img, vis_color=True)
    plt.imshow(img)
    xx, xt= loaded["camera_coords"], loaded["target_coords"]
    plt.scatter(xx[:,0], xx[:,2], zorder=100, color='r', s=0.5)
    for i in range(len(xx)):
        plt.plot(   [xx[i,0],  xt[i,0]],
                    [xx[i,1],  xt[i,1]])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.show()
    print("intrinsic", loaded["intrinsic"])
    return plt.gcf()
def filter_straight(dset_dir="output", 
                    sequence="2013_05_28_drive_0000_sync", 
                    traj_i=100):
    
    loaded = np.load("%s/labels/%s_%08d/boxes.npz" % (dset_dir,sequence, traj_i))
    xx, xt= loaded["camera_coords"], loaded["target_coords"]
    


#%%
import glob
def get_all_campos(dset_dir="output/kitti360_v1", 
                    sequence="2013_05_28_drive_0000_sync", 
                ):
    fses = glob.glob("%s/labels/%s_*/boxes.npz" % (dset_dir,sequence))
    total_N = len(fses)
    xxs,xts = [],[]
    has_empty_semantics = np.zeros(total_N, dtype=bool)
    for i in range(total_N):
        loaded = np.load("%s/labels/%s_%08d/boxes.npz" % (dset_dir,sequence, i))
        xx, xt= loaded["camera_coords"], loaded["target_coords"]
        xxs.append(xx)
        xts.append(xt)
        _, hs,vs = loaded["layout"].shape # check the middle vertical line
        # if the line is all "black", then mark it as empty
        #if_empty = (loaded["layout"][:, vs//2 ] == 0).sum() > hs//2
        if_empty = (loaded["layout"][0, -1, vs//2 ] == 0)
        has_empty_semantics[i] = if_empty
    xxs = np.stack(xxs, axis=0)
    xts = np.stack(xts, axis=0)
    return xxs, xts, has_empty_semantics
def get_campos_filters(xxs, xts, has_empty_semantics, sequence):
    infostr=""
    filters={}

    fltr_name = "full"
    fltr = np.ones(len(xxs), dtype=bool)
    filters[fltr_name] = fltr
    infostr = infostr + ("INFO: there are %d camposes in total\n" % (fltr.sum() ))

    out_the_box = np.logical_or(    xxs.min(axis=-1) < -1., 
                                    xxs.max(axis=-1) >  1. )
    out_the_box = np.any(out_the_box, axis=-1)
    fltr = np.logical_not(out_the_box)
    fltr = np.logical_and(fltr, np.logical_not(has_empty_semantics))
    if sequence == "2013_05_28_drive_0002_sync":
        fltr[:4114] = False # this sequence has a lot of empty perspectives
        # and the first 4114 frames are all empty
        # the 4114 is determined by checking cam0_to_world.txt
        # The first perp image is at frame 4391
        # which corresponds to the 4114th camera traj in the cam0_to_world.txt
    fltr_name = "basic_filter"
    infostr = infostr + ("%s pass ratio %.2f\n" % (fltr_name, fltr.sum() / len(fltr)))
    filters[fltr_name] = fltr
    basic_filter = fltr

    #xxs, xts = xxs[~out_the_box], xts[~out_the_box]
    dirs = (xts - xxs) / np.linalg.norm((xts - xxs), axis=-1, keepdims=True)
    dot = (dirs * np.array([0,1,0.])[None,None,:]).sum(axis=-1)


    fltr_name = "std_direction_filter"
    fltr = np.all(dot>np.cos(np.pi/6.), axis=-1)
    infostr = infostr + ("%s (30 deg) pass ratio %.2f\n" % (fltr_name, fltr.sum() / len(fltr)))
    filters[fltr_name] = fltr

    fltr_name = "std_center_filter"
    fltr = np.all(np.abs(xxs[:,:,0]) < 0.3, axis=-1)
    infostr = infostr + ("%s (±.3) pass ratio %.2f\n" % (fltr_name, fltr.sum() / len(fltr)))
    filters[fltr_name] = fltr

    fltr_name = "std_filter"
    fltr = np.logical_and(filters["std_center_filter"], filters["std_direction_filter"])
    fltr = np.logical_and(basic_filter, fltr)
    infostr = infostr + ("%s pass ratio %.2f\n" % (fltr_name, fltr.sum() / len(fltr)))
    filters[fltr_name] = fltr


    fltr_name = "strict_direction_filter"
    fltr = np.all(dot>np.cos(np.pi/12.), axis=-1)
    infostr = infostr + ("%s (15 deg) pass ratio %.2f\n" % (fltr_name, fltr.sum() / len(fltr)))
    filters[fltr_name] = fltr

    fltr_name = "strict_center_filter"
    fltr = np.all(np.abs(xxs[:,:,0]) < 0.1, axis=-1)
    infostr = infostr + ("%s (±.1) pass ratio %.2f\n" % (fltr_name, fltr.sum() / len(fltr)))
    filters[fltr_name] = fltr

    fltr_name = "strict_filter"
    fltr = np.logical_and(filters["strict_center_filter"], filters["strict_direction_filter"])
    fltr = np.logical_and(basic_filter, fltr)
    infostr = infostr + ("%s pass ratio %.2f\n" % (fltr_name, fltr.sum() / len(fltr)))
    filters[fltr_name] = fltr

    num = None
    for filt in filters:
        if num is None:
            num = filters[filt].shape
        assert num == filters[filt].shape
    return filters, infostr
def seq_export_filters(dset_dir="output/kitti360_v1", 
                    sequence="2013_05_28_drive_0000_sync"):
    xxs, xts, has_empty_semantics = get_all_campos(dset_dir=dset_dir, sequence=sequence)
    filters, infostr = get_campos_filters(xxs, xts, has_empty_semantics, sequence)
    print(infostr)
    sysutil.mkdir("%s/filters"%(dset_dir))
    np.savez("%s/filters/%s.npz"%(dset_dir, sequence), **filters)
    with open("%s/filters/info_%s.txt"%(dset_dir, sequence), "w") as f:
        f.write(infostr)
    return filters, infostr
# %%
def combine_all_sequence_filters(dset_dir="output/kitti360_v1"):
    dsetfilts = None
    df_dict = {}
    for sequence in kitti360_sequences:
        filt = np.load("%s/filters/%s.npz"%(dset_dir, sequence))
        if dsetfilts is None:
            dsetfilts = {k:[] for k in filt}
            df_dict   = {k:{} for k in filt}
        for k in filt:
            dsetfilts[k].append(filt[k])
            print(filt[k].shape)
            for i in range(filt[k].shape[0]):
                name = "%s_%08d"%(sequence, i)
                df_dict[k][name] = filt[k][i]
    for k in dsetfilts:
        dsetfilts[k] = np.concatenate(dsetfilts[k])
    np.savez("%s/filters/dset_array.npz"%(dset_dir), **dsetfilts)
    np.savez("%s/filters/dset_dict.npz"%(dset_dir), **df_dict)
    # generate infostr, first give the number of camposes
    infostr = "INFO: there are %d camposes in total\n" % (dsetfilts[k].shape[0])

    for k in dsetfilts:
        infostr += "%s pass ratio %.2f\n" % (k, dsetfilts[k].sum() / len(dsetfilts[k]))
    with open("%s/filters/info_dset.txt"%(dset_dir), "w") as f:
        f.write(infostr)
if __name__ == "__main__":
    
    combine_all_sequence_filters(dset_dir="output/kitti360_v1")
        
    
    # for sequence in kitti360_sequences[0:1]:
    #     if "0002" not in sequence:
    #         continue
    #     #sequence = "2013_05_28_drive_0003_sync"
    #     print("sequence", sequence)
    #     filters, _ = seq_export_filters(dset_dir="output/kitti360_v0", sequence=sequence)
    #     for filter_name in filters:
    #         filters[filter_name] = filters[filter_name]
    #     kept_names = ["full", "basic_filter", "std_filter", "strict_filter"]
    #     kept_filters = {k:filters[k] for k in kept_names}
    #     export_datavis_videos(total_trajs=kept_filters["full"].sum(), fps=120, dset_dir="output/kitti360_v1", filters=kept_filters, sequence=sequence)
    
    #test_load()
    #export_ditem(traj_i=100)
    #vis_video(trajs=10183, fps=50, dset_dir="output3")
    #visualize_ditem(dset_dir="output3", traj_i=130)
    #test_ditem(dset_dir="output3", traj_i=520)
    #plt.savefig('temp/%04d.png'%130)
    # for i in range(10,100):
    #     visualize_ditem(dset_dir="output",traj_i=i*10)
    #     plt.savefig('temp/%04d.png'%i)
    #     plt.clf()
    #     plt.show()
    #plt.show()
    #plt.savefig("test.png")
# %%