# %%
import os
KITTI_root = "../KITTI-360"
if not os.path.exists(KITTI_root):
    os.mkdir(KITTI_root)
os.chdir(KITTI_root)
downloads_root = os.path.join(KITTI_root, "downloads")
if not os.path.exists( downloads_root ):
    os.mkdir(downloads_root)
# %%
os.chdir(downloads_root)
os.system("wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_2d_perspective.zip")
os.system("wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ffa164387078f48a20f0188aa31b0384bb19ce60/data_3d_bboxes.zip")
os.system("wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip")
os.system("wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip")

os.system("unzip calibration.zip -d %s/calibration" % KITTI_root)
os.system("unzip download_2d_perspective.zip -d %s" % downloads_root)
os.system("bash download_2d_perspective.sh -d %s" % downloads_root)
os.system("mv %s/KITTI-360/data_2d_raw %s" % (KITTI_root, KITTI_root))

os.chdir(KITTI_root)
# %%
os.system("mv data_3d_bboxes/train 3d_bboxes/")
os.system("mv data_3d_bboxes/train_full 3d_bboxes_full/")

# %%
