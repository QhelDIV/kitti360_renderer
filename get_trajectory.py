
# %%
from open3d.web_visualizer import draw
import fresnelvis as fvis
import open3d as o3d
import matplotlib.lines as mlines
import xgutils.vis.fresnelvis as fvis
from PIL import Image
from kitti360_dataset import SequenceProcessor
import kitti360_dataset
from labels import labels as kitti_labels
from labels import id2label, kittiId2label, name2label
import glob
import sys
from utils import Annotation3D_fixed
from kitti360scripts.helpers.annotation import Annotation3D
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
%load_ext autoreload
%autoreload 2
# %%[markdown]
# aobj.R and aobj.T are in
#from xgutils.vis import *
#import open3d
#import open3d as o3d
#from open3d.web_visualizer import draw
# %%

kitti360_root = "/localhome/xya120/studio/sherwin_project/KITTI-360"
sequence = "2013_05_28_drive_0000_sync"
sequence_processor = SequenceProcessor(kitti360_root, sequence)
# sequence_processor.overview_plot(vscale=300)

#img = sequence_processor.global_plot()
# %%
# sequence_processor.setup_visualizer(solid=0.)
sequence_processor.setup_visualizer(solid=1.)
imgs = []
for i in range(0, 1):
    traj_i = 100+i*50  # i*10 +5050 #+ 20*2 + 24*30 + 6485
    sequence_processor.setup_traj(traj_i=traj_i)
    # sequence_processor.show_traj()
    sequence_processor.enable_debug()
    sequence_processor.get_persp_img(traj_i=traj_i, crop_size=None)
    sequence_processor.get_persp_img(traj_i=traj_i, crop_size=256)
    #sequence_processor.perspect_plot_matplotlib(traj_i=traj_i)
    # sequence_processor.get_persp_img(traj_i=traj_i)
    #sequence_processor.get_persp_img(traj_i=traj_i, if_rectified=False)
    plt.show()
    #img = sequence_processor.perspect_plot(
    #    vscale=1, traj_i=traj_i, if_rectified=True)
    img = sequence_processor.topview_plot(
        vscale=50, traj_i=traj_i, hide_frustum=False, mode="bottom", hide_vegetation=True)
    #img = sequence_processor.topview_plot(vscale=50, traj_i=traj_i, hide_vegetation=True, hide_frustum=False)

    #sequence_processor.zoomout_plot(vscale=150, traj_i=traj_i
    sequence_processor.unset_traj()
    # imgs.append(img)
    # save image using PIL:
    #img = Image.fromarray(img)
    #img.save('temp/traj_%d.png' % traj_i)

# %%q
np.set_printoptions(3)
loaded = np.load("data/bedrooms_boxes.npz")
#print(loaded["camera_coords"].shape, loaded["target_coords"].shape, loaded["room_layout"].shape)

#print(loaded["target_coords"] - loaded["camera_coords"])
# %%

kitti360_root = "/localhome/xya120/studio/sherwin_project/KITTI-360"
sequence = "2013_05_28_drive_0000_sync"
sys.path.append(kitti360_root)
poses_data = np.loadtxt("%s/data_poses/%s/poses.txt" %
                        (kitti360_root, sequence))

np.set_printoptions(3)
np.set_printoptions(suppress=True)

poses_matrices = poses_data[:, 1:].reshape(-1, 3, 4)
# %%

#labelDir = '/localhome/xya120/studio/sherwin_project/KITTI-360/data_3d_bboxes/'
labelDir = '%s/3d_bboxes_full/' % kitti360_root
#labelDir = '%s/3d_bboxes_full/' % kitti360_root
labelPath = glob.glob(os.path.join(labelDir, '*', '%s.xml' % sequence))
#print(os.path.join(labelDir, '*', '%s.xml' % sequence), labelPath)
#annon = Annotation3D(labelDir=labelDir, sequence=sequence)
annon = Annotation3D_fixed(labelDir=labelDir, sequence=sequence)

kmeshes = []
centers = []
aobjtrans = []
labels = []
for jnd in list(annon.objects.keys())[:]:
    for ind in annon.objects[jnd].keys():
        aobj = annon.objects[jnd][ind]
        # if id2label[aobj.semanticId].name != 'road':
        #    continue
        vert, face = aobj.vertices, aobj.faces.astype(np.int32)
        aobjtrans.append(np.concatenate([aobj.R, aobj.T[:, None]], axis=1))
        labels.append(aobj.semanticId)
        #vert = (aobj.vertices-aobj.T) @ aobj.R + aobj.T
        kmeshes.append([vert, face])
aobjtrans = np.array(aobjtrans)
labels = np.array(labels)
#centers = np.array(centers)
#center = centers.mean(axis=0)
# for i in range(centers.shape[0]):
#    meshes[i].translate(-center)
# print(center)

# %%

#from fvis import *


def testscene(meshes, pcverts=None, vscale=10, lookat=np.array([0., 0., 0.]), camPos=np.array([0, 2, 0])):
    camera_kwargs = dict(camPos=np.array([2, 2, 2])*vscale, camLookat=np.array([0., 0., 0.]),
                         camUp=np.array([0, 1, 0]), camHeight=2*vscale, fit_camera=False,
                         light_samples=32, samples=32, resolution=(256, 256))

    renderer = fvis.FresnelRenderer(camera_kwargs=camera_kwargs)
    for vert, face in meshes:
        renderer.add_mesh(vert, face, color=np.random.rand(3))
    #renderer.add_cloud(pcverts, radius=3)
    img = renderer.render(preview=True)
    return img


meshes = []
verts = []
for vert, face in kmeshes[:]:
    vert = vert
    meshes.append([vert, face])
    verts.append(vert)
verts = np.concatenate(verts, axis=0)
for i in range(len(meshes)):
    meshes[i][0] = meshes[i][0]  # - verts.mean(axis=0)
print(verts.max(axis=0), verts.min(axis=0), verts.mean(axis=0))
# img = testscene(meshes, pcverts=verts, lookat=meshes[0][0].means(axis=0)*np.array([[1,0,1.]]),
#                        camPos=verts.mean(axis=0))
# img = testscene(meshes, pcverts=verts, lookat=meshes[0][0].mean(axis=0),
#                        camPos=meshes[0][0].mean(axis=0)+np.array([15,15,15]), vscale=200)
# plt.imshow(img)


vscale = 80
cter = (verts.max(axis=0) + verts.min(axis=0))/2
ater = (poses_matrices[:, :, 3].max(axis=0) +
        poses_matrices[:, :, 3].min(axis=0))/2
camera_kwargs = dict(camPos=ater+np.array([0, 0, 150]),
                     camLookat=ater,
                     camUp=-poses_matrices[0, :3, 2],
                     camHeight=2*vscale,
                     fit_camera=False, light_samples=32, samples=32,
                     resolution=np.array((256, 256))*4
                     )

renderer = fvis.FresnelRenderer(camera_kwargs=camera_kwargs)
for i, (vert, face) in enumerate(meshes[:]):
    color = np.array(id2label[labels[i]].color)/255
    renderer.add_mesh(vert, face, color=color, solid=1.)
renderer.add_cloud(aobjtrans[:, :, 3], radius=0.3, color=[0, 1, 0])
renderer.add_cloud(poses_matrices[:, :, 3], radius=.6, color=[1, 0, 0])
img = renderer.render(preview=True)
plt.imshow(img)

ulbs, ucounts = np.unique(labels, return_counts=True)
ulbs, ucounts = zip(*sorted(zip(ulbs, ucounts), key=lambda x: -x[1]))
ulbs = [id2label[ulb] for ulb in ulbs]
colorlegends = [mlines.Line2D([], [], color=np.array(lg.color)/256., marker='s', linestyle='None',
                              markersize=10, label=lg.name) for lg in ulbs]
colorlegends = [mlines.Line2D([], [], color=[1, 0, 0], marker='.', linestyle='None',
                              markersize=10, label='camera')] + colorlegends
plt.legend(handles=colorlegends, loc='upper left', prop={'size': 6},
           bbox_to_anchor=(1.04, 1))
# remove the ticks
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.show()
# matrices[:,:,3]

m = meshes[0][0].mean(axis=0)
# %%
ulbs, ucounts = np.unique(labels, return_counts=True)
ulbs, ucounts = zip(*sorted(zip(ulbs, ucounts), key=lambda x: x[1]))
ulbs = [id2label[ulb] for ulb in ulbs]
colorlegends = [mlines.Line2D([], [], color=np.array(lg.color)/256., marker='s', linestyle='None',
                              markersize=10, label=lg.name) for lg in ulbs]
colorlegends = [mlines.Line2D([], [], color=[1, 0, 0], marker='.', linestyle='None',
                              markersize=10, label='camera')] + colorlegends
plt.legend(handles=colorlegends, loc='upper left', prop={'size': 6},
           bbox_to_anchor=(1.04, 1))
# turn off the ticks
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.show()


# %%
eye = meshes[0].get_center()
lookat = meshes[1].get_center()
print(eye, lookat)
# , eye=meshes[0].get_center(), lookat=meshes[1].get_center(), up=[0,1,0])
draw([*meshes])
# %%

# Load the 3D object
mesh = meshes[0]

# Define the camera intrinsic parameters
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# Define the camera extrinsic parameter, the view point is facing -Z direction
extrinsic = o3d.geometry.Pose()
extrinsic.look_at([0, 0, 0], [0, 0, -1], [0, 1, 0])

# Get the top-view projection
rgb_image, depth_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
    mesh, intrinsic, extrinsic, depth_scale=1.0, depth_trunc=1000, convert_rgb_to_intensity=False)

# The projection is in form of 2D image, you can visualize it using openCV or matplotlib

# %%
#from fvis import *
img = fvis.render_test_scene()
plt.imshow(img)
# %%


points = (np.random.rand(1000, 3) - 0.5) / 4
colors = np.random.rand(1000, 3)


# %%
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(matrices[:, :, 3])
#pc.normals = o3d.utility.Vector3dVector(np.random.rand(matrices.shape[0],3))
# cube_red.compute_vertex_normals()
# cube_red.paint_uniform_color((1.0, 0.0, 0.0))\
#visualizer = JVisualizer()
# visualizer.add_geometry(pc)
print(help(o3d.web_visualizer.draw))
o3d.web_visualizer.draw(pc)
# o3d.visualization.draw_geometries([pc])
# draw(cube_red)
# %%

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
# %%

# define the batch of matrices to be multiplied
matrix_batch = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])

# define the matrix to be multiplied with the batch
multiplier = np.array([[2, 0], [0, 2]])

# perform the batch matrix multiplication
result = np.matmul(matrix_batch, multiplier)

print(result)
# %%
# trash code

#centers.append( vert.mean(axis=0) )
#vert, face = open3d.utility.Vector3dVector(vert), open3d.utility.Vector3iVector(aobj.faces)
#mesh = open3d.geometry.TriangleMesh(vertices=vert, triangles=face)
# mesh.compute_vertex_normals()
# mesh.paint_uniform_color(list(np.random.rand(3)))
