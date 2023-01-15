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
        self.fvis_traj = renderer.add_cloud(poses_matrices[:,:3,3], radius=.6, color=[1,0,0], solid=solid)

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
        framei = self.frames[traj_i]
        checker = np.logical_and(self.timestamps != framei, self.timestamps != -1)
        self.fvis_obj_visibilities[checker] = False
        for obj_i in np.where(checker==True)[0]:
            self.fvis_obj_meshes[obj_i].disable()
    def unset_traj(self):
        for obj_i in np.where(self.fvis_obj_visibilities==False)[0]:
            self.fvis_obj_meshes[obj_i].enable()

    def zoomout_plot(self, vscale=200, traj_i=2):
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
    