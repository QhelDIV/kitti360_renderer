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
        self.poses_data = np.loadtxt("%s/data_poses/%s/poses.txt" % (kitti360_root, sequence))
        self.poses_matrices = self.poses_data[:,1:].reshape(-1, 3, 4)
        self.labelDir = '%s/3d_bboxes_full/' % kitti360_root
        self.annon = Annotation3D_fixed(labelDir=self.labelDir, sequence=sequence)
        self.kmeshes = []
        self.centers = []
        self.aobjtrans=[]
        self.labels = []
        for jnd in list(self.annon.objects.keys())[:]:
            for ind in self.annon.objects[jnd].keys():
                aobj = self.annon.objects[jnd][ind]
                vert, face = aobj.vertices, aobj.faces.astype(np.int32)
                self.aobjtrans.append(np.concatenate([aobj.R, aobj.T[:,None]], axis=1))
                self.labels.append(aobj.semanticId)
                #vert = (aobj.vertices-aobj.T) @ aobj.R + aobj.T
                self.kmeshes.append([vert, face])
        self.aobjtrans = np.array(self.aobjtrans)
        self.labels = np.array(self.labels)
        self.setup_visualizer()
    
    def setup_visualizer(self, solid=1.):
        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        self.renderer = renderer = fvis.FresnelRenderer()
        for i, (vert, face) in enumerate(kmeshes[:]):
            color = np.array(id2label[labels[i]].color)/255
            renderer.add_mesh(vert, face, color= color, solid=solid )
        renderer.add_cloud(aobjtrans[:,:,3], radius=0.3, color=[0,1,0], solid=solid)
        renderer.add_cloud(poses_matrices[:,:,3], radius=.6, color=[1,0,0], solid=solid)

        ulbs, ucounts = np.unique(labels, return_counts=True)
        ulbs, ucounts = zip(*sorted(zip(ulbs,ucounts), key=lambda x: -x[1]))
        ulbs = [id2label[ulb] for ulb in ulbs]
        colorlegends =     [mlines.Line2D([], [], color=np.array(lg.color)/256., marker='s', linestyle='None',
                                    markersize=10, label=lg.name) for lg in ulbs]
        colorlegends=   [mlines.Line2D([], [], color=[1,0,0], marker='.', linestyle='None',
                                    markersize=10, label='camera')] \
                        + colorlegends
        self.colorlegends = colorlegends
    def test_plot(self, vscale=180, ith=2):
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        ater = (poses_matrices[:,:,3].mean(axis=0)) # + poses_matrices[:,:,3].min(axis=0))/2
        #camUp = poses_matrices[0,2,:3]
        camLookat = poses_matrices[ith,:3,3]
        camUp = -poses_matrices[ith,:3,2]
        camera_kwargs = dict(   camPos=camLookat + camUp*10, 
                                camLookat=camLookat,\
                                camUp=camUp,
                                camHeight=2*vscale, 
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
        # ax.axis('off')
        plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
    def overview_plot(self, vscale=180):
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        ater = (poses_matrices[:,:,3].max(axis=0) + poses_matrices[:,:,3].min(axis=0))/2
        camera_kwargs = dict(   camPos=ater + np.array([0,150,150]),#- poses_matrices[0,:3,2]*100, 
                                camLookat=ater,\
                                camUp=-poses_matrices[0,:3,2],
                                camHeight=2*vscale, 
                                fit_camera=False, light_samples=32, samples=32, 
                                resolution=np.array((256,256))*4
                                )
        self.renderer.setup_camera(camera_kwargs)
        ith = np.arange(0, poses_matrices.shape[0], 50)
        camLookat = poses_matrices[ith,:3,3]
        camUp = -poses_matrices[ith,:3,2]
        camPos=camLookat + camUp*100
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
    