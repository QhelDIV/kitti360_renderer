# Challenges of KITTI360 dataset
- labels are outdated and missing
- not all tracks has 3d bboxes
- some tracks are not covered by the 3d bboxes (only partially having bboxes)

# Notes

- interpolate cameras
- disable anti-aliasing for render in order to map the color back to label

# Processed dataset
Inside dataset directory, there are several folders:
- `images/$sequence_name$_$frameid$/0000.png`: 2D center-cropped 256x256 perspective images
- `labels/$sequence_name$_$frameid$/0000.png`: Various labels
    - **camera_coords**: (Nx40x3) camera coordinates. For each camera position (which is regarded as a individual scene), we generate 40 sampled camera positions. The camera position is sampled according to the real camera trajectory. For more details please refer to later section. Also, for the last channel, x, y, z stands for right, forward, up respectively. 
    - **target_coords**: the lookat point of the camera, in camera coordinates
    - **intrinsic**: (3x3 matrix) normalized intrinsic matrix, see the next section for details
    - **denormed_intrinsic**: (3x3 matrix) denormalized intrinsic matrix (denormalized to 256x256)
    - **layout**: (256x256) layout label, see `labels.py` for label meanings
    - **layout_noveg**: (256x256) layout label without vegetation, since many occulusion happens with vegetation
- `filters/$sequence_name$.npz`: the zipped npys of different filters, see the filtering section for details. 
- `datavis/$filter_name$_$sequence_name$`: visualization of the dataset

## Intrinsic matrix
The normalization algorithm is as follows:

    def normalize_intrinsics(K, shape):
        K = K.copy()
        K[0, 2] /= shape[0]
        K[1, 2] /= shape[1]
        K[0, 0] /= shape[1]
        K[1, 1] /= shape[1]
        return K
    def denormalize_intrinsics(K, shape):
        K = K.copy()
        K[0, 2] *= shape[0]
        K[1, 2] *= shape[1]
        K[0, 0] *= shape[1]
        K[1, 1] *= shape[1]
        return K
**Note that the shape is (width, height)**

The principal point is scaled with both width and height, while the focal length is scaled with height only. This is because the principal point is in the center of the image, and the focal length should be the same for both width and height.
## Generating the sampled camera poses for each scene


## Filtering the camera poses
Since the sampled camera poses maybe very challenging: going out of the scene box, turn around or even go back, we filter the camera poses by criterias of different strictness. The criteria is as follows:

- **basic_filter**: the camera should be inside the scene box
- **std_filter**: **basic_filter** + the facing direction should not deviate more than **30** degrees from the original camera facing direction + the sampled camera position should stay in the middle of the scene box (**±.3**)
- **strict_filter**: **std_filter** with stricter criteria: the facing direction should not deviate more than **15** degrees from the original camera facing direction + the sampled camera position should stay in the middle of the scene box (**±.1**)

## Log

This version (v1) fixed previous errors in the v0 dataset.
1. The way to crop the rgb image is corrected. Previously, we directly center-cropped the image, which is not correct. Now we crop with maximum height and then rescale the image to 256x256.
2. The intrinsics are corrected and valided by rendering alignment.
3. The sampled 40 cameras are corrected (see the video)
4. We created the filtering list of the dataset. Please check the README.md in the dataset.
5. We processed all of the sequences in the dataset.