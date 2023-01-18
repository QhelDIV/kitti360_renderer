import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#from kitti360scripts.helpers.annotation import Annotation3D

import fresnelvis as fvis
from utils import Annotation3D_fixed
from labels import id2label, kittiId2label, name2label
from labels import labels as kitti_labels
from kitti360scripts.devkits.commons.loadCalibration import loadPerspectiveIntrinsic

class SequenceProcessor():
    def __init__(self, kitti360_root, sequence):
        self.kitti360_root = kitti360_root
        self.sequence = sequence
        #self.poses_data = np.loadtxt("%s/data_poses/%s/poses.txt" % (kitti360_root, sequence))
        # cam0_to_world.txt is perspective camera 0 to world, x = right, y = down, z = forward
        self.poses_data = np.loadtxt("%s/data_poses/%s/cam0_to_world.txt" % (kitti360_root, sequence))
        self.frames = self.poses_data[:,0]
        self.frame2ind = {frame:ind for ind, frame in enumerate(self.frames)}
        self.poses_matrices = self.poses_data[:,1:].reshape(-1, 4, 4)
        self.labelDir = '%s/3d_bboxes_full/' % kitti360_root
        self.perspImg0Dir = '%s/data_2d_raw/%s/image_00/' % (kitti360_root, sequence)
        self.cam_calib = loadPerspectiveIntrinsic("%s/calibration/perspective.txt" % (kitti360_root))
        self.annon = Annotation3D_fixed(labelDir=self.labelDir, sequence=sequence)
        self.kmeshes = []
        self.centers = []
        self.aobjtrans=[]
        self.labels = []
        self.timestamps = []
        for jnd in list(self.annon.objects.keys())[:]:
            for ind in self.annon.objects[jnd].keys():
                aobj = self.annon.objects[jnd][ind]
                vert, face = aobj.vertices, aobj.faces.astype(np.int32)
                self.aobjtrans.append(np.concatenate([aobj.R, aobj.T[:,None]], axis=1))
                self.labels.append(aobj.semanticId)
                #vert = (aobj.vertices-aobj.T) @ aobj.R + aobj.T
                self.kmeshes.append([vert, face])
                self.timestamps.append( aobj.timestamp )
        self.is_vegetation = np.array([
                id2label[lb].name == "vegetation"
                    for lb in self.labels])

        self.timestamps = np.array(self.timestamps)
        self.aobjtrans = np.array(self.aobjtrans)
        self.labels = np.array(self.labels)
        self.setup_visualizer()
    
    def setup_visualizer(self, solid=1.):
        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        self.renderer = renderer = fvis.FresnelRenderer()
        self.fvis_obj_meshes = []
        self.fvis_obj_visibilities = np.ones(len(kmeshes), dtype=np.bool)
        for i, (vert, face) in enumerate(kmeshes[:]):
            color = np.array(id2label[labels[i]].color)/255
            mesh = renderer.add_mesh(vert, face, color= color, solid=solid )
            self.fvis_obj_meshes.append(mesh)
        #renderer.add_cloud(aobjtrans[:,:,3], radius=0.3, color=[0,1,0], solid=solid)
        self.fvis_traj = renderer.add_cloud(poses_matrices[:,:3,3], radius=.3, color=[1,0,0], solid=solid)
        self.hide_traj()
        #self.fvis_traj = fvis.addBBox(renderer.scene, poses_matrices[:,:3,3], radius=.3, color=[1,0,0], solid=solid)
        # renderer.add_cloud(poses_matrices[:,:3,3], radius=.3, color=[1,0,0], solid=solid)

        ulbs, ucounts = np.unique(labels, return_counts=True)
        ulbs, ucounts = zip(*sorted(zip(ulbs,ucounts), key=lambda x: -x[1]))
        ulbs = [id2label[ulb] for ulb in ulbs]
        colorlegends =     [mlines.Line2D([], [], color=np.array(lg.color)/256., marker='s', linestyle='None',
                                    markersize=10, label=lg.name) for lg in ulbs]
        colorlegends=   [mlines.Line2D([], [], color=[1,0,0], marker='.', linestyle='None',
                                    markersize=10, label='camera')] \
                        + colorlegends
        self.colorlegends = colorlegends
    def setup_traj(self, traj_i):
        """remove duplicate dynamic objects at other frames"""
        framei = self.frames[traj_i]
        checker = np.logical_and(self.timestamps != framei, self.timestamps != -1)
        self.fvis_obj_visibilities[checker] = False
        for obj_i in np.where(checker==True)[0]:
            self.fvis_obj_meshes[obj_i].disable()
        
        self.traj_mesh = self.renderer.add_cloud(self.poses_matrices[traj_i:traj_i+1,:3,3], color= np.array([[1.,1.,0.]]), 
        radius=.6, solid=1. )
        self.traj_mesh.disable()

        frustum_verts = np.array([])
        self.traj_frustum = fresnel.geometry.Polygon(self.renderer.scene,
                                    N=3,
                                    vertices = frustum_verts)
        geometry.material.color = fresnel.color.linear([1.,1.,0.])
        geometry.material.solid=1

        
    def unset_traj(self):
        for obj_i in np.where(self.fvis_obj_visibilities==False)[0]:
            self.fvis_obj_meshes[obj_i].enable()
    def hide_vegetation(self):
        self.fvis_obj_visibilities[self.is_vegetation] = False
        for obj_i in np.where(self.is_vegetation)[0]:
            self.fvis_obj_meshes[obj_i].disable()
    def show_vegetation(self):
        self.fvis_obj_visibilities[self.is_vegetation] = True
        for obj_i in np.where(self.is_vegetation)[0]:
            self.fvis_obj_meshes[obj_i].enable()
    def hide_traj(self):
        self.fvis_traj.disable()
    def show_traj(self):
        self.fvis_traj.enable()
    def get_persp_img(self, traj_i):
        framei = self.frames[traj_i]
        img = plt.imread(self.perspImg0Dir + 'data_rect/%010d.png' % framei)
        plt.imshow(img)
        return img
    def perspect_plot():
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        ater = (poses_matrices[:,:3,3].mean(axis=0)) # + poses_matrices[:,:,3].min(axis=0))/2
        #camUp = -poses_matrices[0,1,:3]
        camLookat = poses_matrices[traj_i,:3,3]
        camUp = poses_matrices[traj_i,:3,2]
        wolrd_up = np.array([0,0,1.])
        camera_kwargs = dict(   camPos=camLookat + wolrd_up*50, 
                                camLookat=camLookat,\
                                camUp=camUp,
                                camHeight=vscale, 
                                fit_camera=False, light_samples=32, samples=32, 
                                resolution=np.array((256,256))*2
                                )
        self.renderer.setup_camera(camera_kwargs)
        img = self.renderer.render(preview=True)
        plt.imshow( img )

        # plt.legend( handles=self.colorlegends, loc='upper left', prop={'size': 10},
        #             bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.axis('off')
        # plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
        return img
    def topview_plot(self, vscale=100, traj_i=2, hide_traj=True, hide_vegetation=False):
        
        if hide_vegetation:
            self.hide_vegetation()
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        ater = (poses_matrices[:,:3,3].mean(axis=0)) # + poses_matrices[:,:,3].min(axis=0))/2
        #camUp = -poses_matrices[0,1,:3]
        camLookat = poses_matrices[traj_i,:3,3]
        camUp = poses_matrices[traj_i,:3,2]
        wolrd_up = np.array([0,0,1.])
        camera_kwargs = dict(   camPos=camLookat + wolrd_up*50, 
                                camLookat=camLookat,\
                                camUp=camUp,
                                camHeight=vscale, 
                                fit_camera=False, light_samples=32, samples=32, 
                                resolution=np.array((256,256))*2
                                )
        self.renderer.setup_camera(camera_kwargs)
        img = self.renderer.render(preview=True)
        plt.imshow( img )

        # plt.legend( handles=self.colorlegends, loc='upper left', prop={'size': 10},
        #             bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.axis('off')
        # plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
        if hide_vegetation:
            self.show_vegetation()

        return img
    def zoomout_plot(self, vscale=200, traj_i=2):
        self.show_traj()
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        ater = (poses_matrices[:,:3,3].mean(axis=0)) # + poses_matrices[:,:,3].min(axis=0))/2
        #camUp = -poses_matrices[0,1,:3]
        camLookat = poses_matrices[traj_i,:3,3]
        camUp = np.array([1.,0,0.]) #-poses_matrices[traj_i,:3,1]
        wolrd_up = np.array([0,0,1.])
        camera_kwargs = dict(   camPos=camLookat + wolrd_up*50, 
                                camLookat=camLookat,\
                                camUp=camUp,
                                camHeight=vscale, 
                                fit_camera=False, light_samples=32, samples=32, 
                                resolution=np.array((256,256))*2
                                )
        self.renderer.setup_camera(camera_kwargs)
        self.traj_mesh.enable()
        img = self.renderer.render(preview=True)
        plt.imshow( img )

        plt.legend( handles=self.colorlegends, loc='upper left', prop={'size': 10},
                    bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.axis('off')
        plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
        self.hide_traj()
        return img

    def global_plot(self):
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        tranlations = poses_matrices[:,:3,3]
        ater = (tranlations.mean(axis=0)) # + poses_matrices[:,:,3].min(axis=0))/2
        #camUp = -poses_matrices[0,1,:3]
        camLookat = ater
        camUp = np.array([1.,0,0.]) #-poses_matrices[traj_i,:3,1]
        wolrd_up = np.array([0,0,1.])
        vscale  = (tranlations.max(axis=0) - tranlations.min(axis=0)).max()*1.3
        camera_kwargs = dict(   camPos=camLookat + wolrd_up*50, 
                                camLookat=camLookat,\
                                camUp=camUp,
                                camHeight=vscale, 
                                fit_camera=False, light_samples=32, samples=32, 
                                resolution=np.array((256,256))*16
                                )
        self.renderer.setup_camera(camera_kwargs)
        img = self.renderer.render(preview=True)
        plt.imshow( img )

        plt.legend( handles=self.colorlegends, loc='upper left', prop={'size': 10},
                    bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        ax.axis('off')
        plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
        return img
                
    def overview_plot(self, vscale=360):
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        ater = (poses_matrices[:,:3,3].max(axis=0) + poses_matrices[:,:3,3].min(axis=0))/2
        #camUp = np.array([1.,0,0.]) #-poses_matrices[traj_i,:3,1]
        camUp = np.array([0,0,1.])
        world_up = np.array([0,0,1.])
        camera_kwargs = dict(   camPos=ater + np.array([0,1500,1500]),#- poses_matrices[0,:3,1]*100, 
                                camLookat=ater,\
                                camUp=camUp,
                                camHeight=vscale, 
                                fit_camera=False, light_samples=32, samples=32, 
                                resolution=np.array((256,256))*4
                                )
        self.renderer.setup_camera(camera_kwargs)
        traj_i = np.arange(0, poses_matrices.shape[0], 50)
        camLookat = poses_matrices[traj_i,:3,3]
        #camUp = -poses_matrices[traj_i,:3,1]
        camUp = np.array([0,0,1.]) #-poses_matrices[traj_i,:3,1]
        camPos=camLookat + world_up*50
        starts, ends = camPos, camLookat
        fvis.addArrows(self.renderer.scene, starts, ends, radius=2.3)
        img = self.renderer.render(preview=True)
        plt.imshow( img )

        plt.legend( handles=self.colorlegends, loc='upper left', prop={'size': 10},
                    bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        # ax.axis('off')
        plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
# "0008" and "0018" only has poses, but no 3d bboxes
kitti360_sequences = ["0000", "0002", "0003", "0004", "0005", "0006", "0007", "0009", "0010"]
kitti360_sequences = ["2013_05_28_drive_%s_sync"%seq for seq in kitti360_sequences]
class DatasetProcessor:
    def __init__(self, kitti360_root, sequences=kitti360_sequences,
    bbox_name="3d_bboxes_full", build_dir="build"):
        self.kitti360_root = kitti360_root
        self.bbox_name = bbox_name
        self.bbox_root = os.path.join(self.kitti360_root, self.bbox_name)
        self.sequences = sequences
        self.build_dir = build_dir
        # create the build dir if not exists
        if not os.path.exists(self.build_dir):
            os.makedirs(self.build_dir)

    def render_legend(self):
        labels = []
        for seq in self.sequences:
            processor = SequenceProcessor(self.kitti360_root, seq)
            labels.append( processor.labels )
        labels = np.concatenate(labels)
        # plot and export legend
        ulbs, ucounts = np.unique(labels, return_counts=True)
        ulbs, ucounts = zip(*sorted(zip(ulbs,ucounts), key=lambda x: -x[1]))
        ulbs = [id2label[ulb] for ulb in ulbs]
        colorlegends =     [mlines.Line2D([], [], color=np.array(lg.color)/256., marker='s', linestyle='None',
                                    markersize=10, label=lg.name) for lg in ulbs]
        colorlegends= [mlines.Line2D([], [], color=[1,0,0], marker='.', linestyle='None',
                                    markersize=10, label='camera')] + colorlegends

        legend = plt.legend( handles=colorlegends, loc='upper left', prop={'size': 10},
                    bbox_to_anchor=(0, 0), ncol=5)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.axis('off')
        export_legend(legend, filename=os.path.join(self.build_dir, "legend.png"))
    def render_global_views_for_all_tracks(self):
        for seq in self.sequences:
            print(seq)
            processor = SequenceProcessor(self.kitti360_root, seq)

            img = processor.global_plot()
            from PIL import Image
            im = Image.fromarray(img)
            im.save(os.path.join(self.build_dir, "globalview_%s.png"%seq))

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
if __name__ == "__main__":
    kitti360_root = "/localhome/xya120/studio/sherwin_project/KITTI-360"
    processor = DatasetProcessor(kitti360_root)
    #processor.render_legend()
    processor.render_global_views_for_all_tracks()