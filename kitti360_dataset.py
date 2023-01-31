import os
import sys
import glob
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#from kitti360scripts.helpers.annotation import Annotation3D
import fresnel
import xgutils.vis.fresnelvis as fvis
from xgutils import sysutil, nputil
from utils import Annotation3D_fixed
from labels import id2label, kittiId2label, name2label
from labels import labels as kitti_labels
from kitti360scripts.devkits.commons.loadCalibration import loadPerspectiveIntrinsic
from kitti360scripts.helpers.project import CameraPerspective


class SequenceProcessor():
    def __init__(self, kitti360_root, sequence):
        self.kitti360_root = kitti360_root
        self.sequence = sequence
        #self.poses_data = np.loadtxt("%s/data_poses/%s/poses.txt" % (kitti360_root, sequence))
        # cam0_to_world.txt is perspective camera 0 to world, x = right, y = down, z = forward
        self.poses_data = np.loadtxt(
            "%s/data_poses/%s/cam0_to_world.txt" % (kitti360_root, sequence))
        self.frames = self.poses_data[:, 0]
        frameN = len(self.frames)
        self.frame2ind = {frame: ind for ind, frame in enumerate(self.frames)}
        self.poses_matrices = self.poses_data[:, 1:].reshape(-1, 4, 4)
        self.imu_poses = np.loadtxt(
            "%s/data_poses/%s/poses.txt" % (kitti360_root, sequence))
        # repeat [0,0,0,1] to match the shape of poses_matrices
        self.imu_mats = np.concatenate([self.imu_poses[:, 1:],
                                        np.repeat(
                                            np.array([[0, 0, 0, 1]]), frameN, axis=0)
                                        ], axis=1).reshape(-1, 4, 4)
        self.CamP = CameraPerspective(root_dir=self.kitti360_root,
                                      seq=self.sequence, cam_id=0)
        self.cam0_unrect = np.matmul(
            self.imu_mats, self.CamP.camToPose[None, ...])

        self.labelDir = '%s/3d_bboxes_full/' % kitti360_root
        self.perspImg0Dir = '%s/data_2d_raw/%s/image_00/' % (
            kitti360_root, sequence)
        self.cam_calib = loadPerspectiveIntrinsic(
            "%s/calibration/perspective.txt" % (kitti360_root))
        # self.persp
        self.annon = Annotation3D_fixed(
            labelDir=self.labelDir, sequence=sequence)
        self.kmeshes = []
        self.centers = []
        self.aobjtrans = []
        self.labels = []
        self.timestamps = []
        for jnd in list(self.annon.objects.keys())[:]:
            for ind in self.annon.objects[jnd].keys():
                aobj = self.annon.objects[jnd][ind]
                vert, face = aobj.vertices, aobj.faces.astype(np.int32)
                self.aobjtrans.append(np.concatenate(
                    [aobj.R, aobj.T[:, None]], axis=1))
                self.labels.append(aobj.semanticId)
                #vert = (aobj.vertices-aobj.T) @ aobj.R + aobj.T
                self.kmeshes.append([vert, face])
                self.timestamps.append(aobj.timestamp)
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
            mesh = renderer.add_mesh(vert, face, color=color, solid=solid)
            self.fvis_obj_meshes.append(mesh)
        #renderer.add_cloud(aobjtrans[:,:,3], radius=0.3, color=[0,1,0], solid=solid)
        self.fvis_traj = renderer.add_cloud(
            poses_matrices[:, :3, 3], radius=.3, color=[1, 0, 0], solid=solid)
        self.hide_traj()
        #self.fvis_traj = fvis.addBBox(renderer.scene, poses_matrices[:,:3,3], radius=.3, color=[1,0,0], solid=solid)
        # renderer.add_cloud(poses_matrices[:,:3,3], radius=.3, color=[1,0,0], solid=solid)

        ulbs, ucounts = np.unique(labels, return_counts=True)
        ulbs, ucounts = zip(*sorted(zip(ulbs, ucounts), key=lambda x: -x[1]))
        ulbs = [id2label[ulb] for ulb in ulbs]
        colorlegends = [mlines.Line2D([], [], color=np.array(lg.color)/256., marker='s', linestyle='None',
                                      markersize=10, label=lg.name) for lg in ulbs]
        colorlegends = [mlines.Line2D([], [], color=[1, 0, 0], marker='.', linestyle='None',
                                      markersize=10, label='camera')] \
            + colorlegends
        self.colorlegends = colorlegends

    def setup_traj(self, traj_i):
        """remove duplicate dynamic objects at other frames"""
        framei = self.frames[traj_i]
        checker = np.logical_and(
            self.timestamps != framei, self.timestamps != -1)
        self.fvis_obj_visibilities[checker] = False
        for obj_i in np.where(checker == True)[0]:
            self.fvis_obj_meshes[obj_i].disable()

        traj_pos = self.poses_matrices[traj_i, :3, 3]
        self.traj_mesh = self.renderer.add_cloud(traj_pos[None, :], color=np.array([[1., 1., 0.]]),
                                                 radius=.6, solid=1.)
        self.traj_mesh.disable()

        self.frustum_verts = np.array([traj_pos,
                                       traj_pos+np.array([0, 10, 0]),
                                       traj_pos+np.array([10, 0, 0]),
                                       ])
        self.frustum_verts = np.array([[0, 0, 0],
                                       [.3, 0, 1],
                                       [-.3, 0, 1],
                                       ])*10
        self.frustum_verts = self.poses_matrices[traj_i,
                                                 :3, :3] @ self.frustum_verts.T + traj_pos[:, None]
        self.frustum_verts = self.frustum_verts.T

        self.traj_frustum = fresnel.geometry.Mesh(self.renderer.scene,
                                                  N=1,
                                                  vertices=self.frustum_verts)
        self.traj_frustum.material.color = fresnel.color.linear([1., 1., 0.])
        self.traj_frustum.material.solid = 1
        self.traj_frustum.disable()

    def unset_traj(self):
        for obj_i in np.where(self.fvis_obj_visibilities == False)[0]:
            self.fvis_obj_meshes[obj_i].enable()
        self.traj_frustum.remove()
        self.traj_mesh.remove()

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

    def get_persp_img(self, traj_i, if_rectified=True, crop_size=256):
        framei = self.frames[traj_i]
        if if_rectified:
            img = plt.imread(self.perspImg0Dir +
                             'data_rect/%010d.png' % framei)
        else:
            img = plt.imread(self.perspImg0Dir + 'data_rgb/%010d.png' % framei)
        if crop_size is not None:
            img = crop_center(img, cropx=crop_size,
                              cropy=crop_size, resize_to=(256, 256))
            print("image size", img.shape)
        self.persp_img_size = img.shape[:2]
        self.image = img
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        plt.margins(0, 0)
        plt.axis('off')
        return img

    def perspect_plot_matplotlib(self, traj_i=0, max_dist=10., cam_id=0):
        from xgutils import geoutil
        frame = self.frames[traj_i]
        points, depths, tpls = [], [], []
        image = self.image
        plt.imshow(image)  # [:, :, ::-1])

        if cam_id == 0 or cam_id == 1:
            camera = CameraPerspective(
                self.kitti360_root, self.sequence, cam_id)
        # fisheye
        elif cam_id == 2 or cam_id == 3:
            camera = CameraFisheye(self.kitti360_root, self.sequence, cam_id)
            print(camera.fi)
        else:
            raise RuntimeError('Invalid Camera ID!')

        annotation3D = self.annon
        for k, v in annotation3D.objects.items():
            if len(v.keys()) == 1 and (-1 in v.keys()):  # show static only
                obj3d = v[-1]
                obj_color = np.array(id2label[obj3d.semanticId].color) / 256.
                dist = (obj3d.vertices[0] - self.poses_matrices[traj_i, :3, 3])
                dist = np.linalg.norm(dist, axis=0)
                if dist > max_dist:
                    continue

                camera(obj3d, frame)
                vertices = np.asarray(obj3d.vertices_proj).T
                points.append(np.asarray(obj3d.vertices_proj).T)
                depths.append(np.asarray(obj3d.vertices_depth))

                sverts = geoutil.sampleMesh(obj3d.vertices.astype(
                    np.float32), obj3d.faces.astype(np.int32), 1000)
                uv, d = camera.project_vertices(sverts, frame)
                mask = np.logical_and(np.logical_and(
                    d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(
                    mask, uv[0] < image.shape[1]), uv[1] < image.shape[0])
                #plt.plot(uv[0][mask], uv[1][mask], 'r.', linewidth=0.3)
                tpls.append((uv[0][mask], uv[1][mask],
                            obj_color, d[mask].mean()))
        for tpl in tpls:
            plt.plot(tpl[0], tpl[1], color=tpl[2], marker='.', linewidth=0.01,
                     markersize=2.0, zorder=100 + 1/(tpl[3]+0.001))

        plt.show()

    def perspect_plot(self, vscale=10, traj_i=0, if_rectified=True):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        if if_rectified == False:
            poses_matrices = self.cam0_unrect
        # + poses_matrices[:,:,3].min(axis=0))/2
        ater = (poses_matrices[:, :3, 3].mean(axis=0))
        #camUp = -poses_matrices[0,1,:3]
        pmat = poses_matrices[traj_i, :3, :4]
        camPos = pmat[:3, 3]
        camLookat = camPos + pmat[:3, 2]*np.array([1, 1, 0.])*100
        camUp = -pmat[:3, 1]
        wolrd_up = np.array([0, 0, 1.])
        camera_kwargs = dict(camPos=camPos,
                             camLookat=camLookat,
                             camUp=camUp,
                             camHeight=1.,
                             # self.persp_img_size[0],
                             focal_length=552.554 / 353,
                             fit_camera=False, light_samples=32, samples=32,
                             resolution=np.array(self.persp_img_size)[::-1],
                             camera_type="perspective"
                             )
        self.renderer.setup_camera(camera_kwargs)
        img = self.renderer.render(preview=True)
        plt.imshow(img)

        # plt.legend( handles=self.colorlegends, loc='upper left', prop={'size': 10},
        #             bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.axis('off')
        # ax.axis('tight')
        # plt.savefig("test.png", bbox_inches = 'tight',
        #     pad_inches = 0)
        # plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
        return img

    def topview_plot(self, vscale=100, traj_i=2, hide_traj=True, hide_vegetation=False, hide_frustum=True,
                     mode="bottom"):

        if hide_vegetation:
            self.hide_vegetation()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        # + poses_matrices[:,:,3].min(axis=0))/2
        ater = (poses_matrices[:, :3, 3].mean(axis=0))
        #camUp = -poses_matrices[0,1,:3]
        camLookat = poses_matrices[traj_i, :3, 3]
        camUp = poses_matrices[traj_i, :3, 2]
        camHeight = vscale

        if mode == "bottom":
            camLookat = camLookat + camUp*camHeight/2.

        wolrd_up = np.array([0, 0, 1.])
        print(- poses_matrices[traj_i, :3, 1])
        camera_kwargs = dict(camPos=camLookat - poses_matrices[traj_i, :3, 1]*50,
                             camLookat=camLookat,
                             camUp=camUp,
                             camHeight=camHeight,
                             fit_camera=False, light_samples=32, samples=32,
                             resolution=np.array((256, 256))*2
                             )
        self.renderer.setup_camera(camera_kwargs)
        img = self.renderer.render(preview=True)
        plt.imshow(img)

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

    def zoomout_plot(self, vscale=200, traj_i=2, legend=False):
        self.show_traj()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        # + poses_matrices[:,:,3].min(axis=0))/2
        ater = (poses_matrices[:, :3, 3].mean(axis=0))
        #camUp = -poses_matrices[0,1,:3]
        camLookat = poses_matrices[traj_i, :3, 3]
        camUp = np.array([1., 0, 0.])  # -poses_matrices[traj_i,:3,1]
        wolrd_up = np.array([0, 0, 1.])
        camera_kwargs = dict(camPos=camLookat + wolrd_up*50,
                             camLookat=camLookat,
                             camUp=camUp,
                             camHeight=vscale,
                             fit_camera=False, light_samples=32, samples=32,
                             resolution=np.array((256, 256))*2
                             )
        self.renderer.setup_camera(camera_kwargs)
        self.traj_mesh.enable()
        self.traj_frustum.enable()
        img = self.renderer.render(preview=True)
        self.traj_mesh.disable()
        self.traj_frustum.disable()
        plt.imshow(img)
        if legend == True:
            plt.legend(handles=self.colorlegends, loc='upper left', prop={'size': 10},
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        tranlations = poses_matrices[:, :3, 3]
        # + poses_matrices[:,:,3].min(axis=0))/2
        ater = (tranlations.mean(axis=0))
        #camUp = -poses_matrices[0,1,:3]
        camLookat = ater
        camUp = np.array([1., 0, 0.])  # -poses_matrices[traj_i,:3,1]
        wolrd_up = np.array([0, 0, 1.])
        vscale = (tranlations.max(axis=0) - tranlations.min(axis=0)).max()*1.3
        camera_kwargs = dict(camPos=camLookat + wolrd_up*50,
                             camLookat=camLookat,
                             camUp=camUp,
                             camHeight=vscale,
                             fit_camera=False, light_samples=32, samples=32,
                             resolution=np.array((256, 256))*16
                             )
        self.renderer.setup_camera(camera_kwargs)
        img = self.renderer.render(preview=True)
        plt.imshow(img)

        plt.legend(handles=self.colorlegends, loc='upper left', prop={'size': 10},
                   bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        ax.axis('off')
        plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()
        return img

    def overview_plot(self, vscale=360):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        kmeshes, poses_matrices, aobjtrans, labels = self.kmeshes, self.poses_matrices, self.aobjtrans, self.labels
        ater = (poses_matrices[:, :3, 3].max(axis=0) +
                poses_matrices[:, :3, 3].min(axis=0))/2
        # camUp = np.array([1.,0,0.]) #-poses_matrices[traj_i,:3,1]
        camUp = np.array([0, 0, 1.])
        world_up = np.array([0, 0, 1.])
        camera_kwargs = dict(camPos=ater + np.array([0, 1500, 1500]),  # - poses_matrices[0,:3,1]*100,
                             camLookat=ater,\
                             camUp=camUp,
                             camHeight=vscale,
                             fit_camera=False, light_samples=32, samples=32,
                             resolution=np.array((256, 256))*4
                             )
        self.renderer.setup_camera(camera_kwargs)
        traj_i = np.arange(0, poses_matrices.shape[0], 50)
        camLookat = poses_matrices[traj_i, :3, 3]
        #camUp = -poses_matrices[traj_i,:3,1]
        camUp = np.array([0, 0, 1.])  # -poses_matrices[traj_i,:3,1]
        camPos = camLookat + world_up*50
        starts, ends = camPos, camLookat
        fvis.addArrows(self.renderer.scene, starts, ends, radius=2.3)
        img = self.renderer.render(preview=True)
        plt.imshow(img)

        plt.legend(handles=self.colorlegends, loc='upper left', prop={'size': 10},
                   bbox_to_anchor=(0, 0), ncol=5)
        # remove the ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        # ax.axis('off')
        plt.gca().set_title('sequence %s - all timeframes overview plot' % self.sequence)
        plt.show()

    def test_output(self, framei, outdir="output/"):
        frame_name = self.sequence + "_%08d" % framei
        imdir = outdir + "images/" + frame_name + "/"
        lbdir = outdir + "labels/" + frame_name + "/"
        sysutil.mkdirs(imdir)
        sysutil.mkdirs(lbdir)

        sequence_processor.setup_traj(traj_i=traj_i)
        sequence_processor.traj_mesh.disable()
        sequence_processor.traj_frustum.disable()
        img = sequence_processor.get_persp_img(traj_i=traj_i, crop_size=256)
        camK = cam_calib["P_rect_00"]
        camK = normalize_intrinsics(camK, persp_img_size)
        camRT = poses_matrices[traj_i, :3, :4]

        from PIL import Image
        im = Image.fromarray(img)
        im.save(imdir+"%04d.png" % 0)

        #img = sequence_processor.perspect_plot(vscale=1, traj_i=traj_i, if_rectified=True)
        img1 = sequence_processor.topview_plot(
            vscale=50, traj_i=traj_i, hide_vegetation=False, mode="bottom")
        img2 = sequence_processor.topview_plot(
            vscale=50, traj_i=traj_i, hide_vegetation=True, mode="bottom")
        sequence_processor.unset_traj()

        np.savez(lbdir+"%04d.npz" % 0, camK=camK, camRT=camRT,
                 camRTs=poses_matrices[:, :3, :4], camKs=cam_calib["P_rect_00"],)

        self.get_persp_img(traj_i=traj_i, crop_size=256)


def normalize_intrinsics(K, shape):
    K = K.copy()
    K[0, :] /= shape[0]
    K[1, 1] /= shape[0]
    K[1, 2] /= shape[1]
    return K


# "0008" and "0018" only has poses, but no 3d bboxes
kitti360_sequences = ["0000", "0002", "0003",
                      "0004", "0005", "0006", "0007", "0009", "0010"]
kitti360_sequences = ["2013_05_28_drive_%s_sync" %
                      seq for seq in kitti360_sequences]


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
            labels.append(processor.labels)
        labels = np.concatenate(labels)
        # plot and export legend
        ulbs, ucounts = np.unique(labels, return_counts=True)
        ulbs, ucounts = zip(*sorted(zip(ulbs, ucounts), key=lambda x: -x[1]))
        ulbs = [id2label[ulb] for ulb in ulbs]
        colorlegends = [mlines.Line2D([], [], color=np.array(lg.color)/256., marker='s', linestyle='None',
                                      markersize=10, label=lg.name) for lg in ulbs]
        colorlegends = [mlines.Line2D([], [], color=[1, 0, 0], marker='.', linestyle='None',
                                      markersize=10, label='camera')] + colorlegends

        legend = plt.legend(handles=colorlegends, loc='upper left', prop={'size': 10},
                            bbox_to_anchor=(0, 0), ncol=5)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.axis('off')
        export_legend(legend, filename=os.path.join(
            self.build_dir, "legend.png"))

    def render_global_views_for_all_tracks(self):
        for seq in self.sequences:
            print(seq)
            processor = SequenceProcessor(self.kitti360_root, seq)

            img = processor.global_plot()
            from PIL import Image
            im = Image.fromarray(img)
            im.save(os.path.join(self.build_dir, "globalview_%s.png" % seq))


def direct_crop(img, cropx, cropy):
    y, x, c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def crop_center(img, cropx, cropy, resize_to=None):
    zimg = ndimage.zoom(img, zoom=(2., 2., 1.))
    cropped = direct_crop(zimg, cropx*2, cropy*2)
    ratio = .5
    if resize_to is None:
        resize_to = zimg.shape/2.
    ratio = np.array(resize_to) / cropped.shape[:2]
    cropped = ndimage.zoom(cropped, zoom=(*ratio, 1.))
    return cropped


def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


if __name__ == "__main__":
    kitti360_root = "/localhome/xya120/studio/sherwin_project/KITTI-360"
    processor = DatasetProcessor(kitti360_root)
    # processor.render_legend()
    processor.render_global_views_for_all_tracks()
