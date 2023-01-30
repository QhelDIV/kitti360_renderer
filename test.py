# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from labels import id2label
from kitti360scripts.helpers.project import CameraPerspective
# if 'KITTI360_DATASET' in os.environ:
#     kitti360Path = os.environ['KITTI360_DATASET']
# else:
#     kitti360Path = os.path.join(os.path.dirname(
#                             os.path.realpath(__file__)), '..', '..')
kitti360Path = "/localhome/xya120/studio/sherwin_project/KITTI-360/"
seq = 0
cam_id = 0
sequence = '2013_05_28_drive_%04d_sync'%seq
# perspective
if cam_id == 0 or cam_id == 1:
    camera = CameraPerspective(kitti360Path, sequence, cam_id)
# fisheye
elif cam_id == 2 or cam_id == 3:
    camera = CameraFisheye(kitti360Path, sequence, cam_id)
    print(camera.fi)
else:
    raise RuntimeError('Invalid Camera ID!')

# 3D bbox
from utils import Annotation3D_fixed as Annotation3D
label3DBboxPath = os.path.join(kitti360Path, '3d_bboxes_full')
annotation3D = Annotation3D(label3DBboxPath, sequence)
# loop over frames
for frame in camera.frames[::100]:
    # perspective
    if cam_id == 0 or cam_id == 1:
        image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect', '%010d.png'%frame)
    # fisheye
    elif cam_id == 2 or cam_id == 3:
        image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rgb', '%010d.png'%frame)
    else:
        raise RuntimeError('Invalid Camera ID!')
    if not os.path.isfile(image_file):
        print('Missing %s ...' % image_file)
        continue


    print(image_file)
    image = cv2.imread(image_file)
    plt.imshow(image[:,:,::-1])


    points = []
    depths = []
    for k,v in annotation3D.objects.items():
        if len(v.keys())==1 and (-1 in v.keys()): # show static only
            obj3d = v[-1]
            if not id2label[obj3d.semanticId].name=='building': # show buildings only
                continue
            camera(obj3d, frame)
            vertices = np.asarray(obj3d.vertices_proj).T
            points.append(np.asarray(obj3d.vertices_proj).T)
            depths.append(np.asarray(obj3d.vertices_depth))
            for line in obj3d.lines:
                v = [obj3d.vertices[line[0]]*x + obj3d.vertices[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
                uv, d = camera.project_vertices(np.asarray(v), frame)
                mask = np.logical_and(np.logical_and(d>0, uv[0]>0), uv[1]>0)
                mask = np.logical_and(np.logical_and(mask, uv[0]<image.shape[1]), uv[1]<image.shape[0])
                plt.plot(uv[0][mask], uv[1][mask], 'r.')

    plt.pause(0.5)
    plt.clf()
#%%