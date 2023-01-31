# %%
from utils import Annotation3D_fixed as Annotation3D
import cv2
import numpy as np
import matplotlib.pyplot as plt
from labels import id2label
from kitti360scripts.helpers.project import CameraPerspective
from xgutils import geoutil
# if 'KITTI360_DATASET' in os.environ:
#     kitti360Path = os.environ['KITTI360_DATASET']
# else:
#     kitti360Path = os.path.join(os.path.dirname(
#                             os.path.realpath(__file__)), '..', '..')
kitti360Path = "/localhome/xya120/studio/sherwin_project/KITTI-360/"
seq = 0
cam_id = 0
sequence = '2013_05_28_drive_%04d_sync' % seq
# perspective
if cam_id == 0 or cam_id == 1:
    camera = CameraPerspective(kitti360Path, sequence, cam_id)
# fisheye
elif cam_id == 2 or cam_id == 3:
    camera = CameraFisheye(kitti360Path, sequence, cam_id)
    print(camera.fi)
else:
    raise RuntimeError('Invalid Camera ID!')


poses_data = np.loadtxt(
    "%s/data_poses/%s/cam0_to_world.txt" % (kitti360Path, sequence))
frames = poses_data[:, 0]
frameN = len(frames)
frame2ind = {frame: ind for ind, frame in enumerate(frames)}
poses_matrices = poses_data[:, 1:].reshape(-1, 4, 4)


# 3D bbox
label3DBboxPath = os.path.join(kitti360Path, '3d_bboxes_full')
annotation3D = Annotation3D(label3DBboxPath, sequence)
# loop over frames
for frame in camera.frames[17:18]:
    # perspective
    if cam_id == 0 or cam_id == 1:
        image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence,
                                  'image_%02d' % cam_id, 'data_rect', '%010d.png' % frame)
    # fisheye
    elif cam_id == 2 or cam_id == 3:
        image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence,
                                  'image_%02d' % cam_id, 'data_rgb', '%010d.png' % frame)
    else:
        raise RuntimeError('Invalid Camera ID!')
    if not os.path.isfile(image_file):
        print('Missing %s ...' % image_file)
        continue

    print(image_file)
    image = cv2.imread(image_file)
    plt.imshow(image[:, :, ::-1])

    points = []
    depths = []
    tpls = []
    for k, v in annotation3D.objects.items():
        if len(v.keys()) == 1 and (-1 in v.keys()):  # show static only
            obj3d = v[-1]
            # if not id2label[obj3d.semanticId].name == 'building':  # show buildings only
            #     continue
            # if not ("Sign" in id2label[obj3d.semanticId].name):
            #    continue

            dist = (obj3d.vertices[0] - poses_matrices[17, :3, 3])
            dist = np.linalg.norm(dist, axis=0)
            if dist > 10:
                continue
            camera(obj3d, frame)
            vertices = np.asarray(obj3d.vertices_proj).T
            points.append(np.asarray(obj3d.vertices_proj).T)
            depths.append(np.asarray(obj3d.vertices_depth))
            # for line in obj3d.lines:
            # v = [obj3d.vertices[line[0]]*x + obj3d.vertices[line[1]]
            #      * (1-x) for x in np.arange(0, 1, 0.01)]
            #(obj3d.vertices, obj3d.faces)
            sverts = geoutil.sampleMesh(obj3d.vertices.astype(
                np.float32), obj3d.faces.astype(np.int32), 1000)
            uv, d = camera.project_vertices(sverts, frame)
            mask = np.logical_and(np.logical_and(
                d > 0, uv[0] > 0), uv[1] > 0)
            mask = np.logical_and(np.logical_and(
                mask, uv[0] < image.shape[1]), uv[1] < image.shape[0])
            #plt.plot(uv[0][mask], uv[1][mask], 'r.', linewidth=0.3)
            tpls.append((uv[0][mask], uv[1][mask], d.mean()))
    for tpl in tpls:
        plt.plot(tpl[0], tpl[1], marker='.', linewidth=0.01,
                 markersize=2.0, zorder=100 + 1/(tpl[2]+0.001))

    plt.pause(0.5)
    plt.clf()
# %%
