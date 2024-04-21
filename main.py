import os
import numpy as np
from tqdm import tqdm
from labels import semantics2rgbimg, kitti360_sequences

from kitti360_processor import DatasetProcessor
import kitti360_dataset

def filter_dataset(dset_dir, out_dir, filter_name):
    # make dir if not exist 
    filters = np.load(f'{dset_dir}/filters/dset_dict.npz', allow_pickle=True)
    filter_dict = filters[filter_name].item()
    if not os.path.exists(out_dir):
        os.mkdir(f'{out_dir}')
        os.mkdir(f'{out_dir}/images')
        os.mkdir(f'{out_dir}/labels')
    print(filter_dict.keys())
    for key in tqdm(filter_dict):
        os.system(f'cp -r {dset_dir}/images/{key} {out_dir}/images/{key}')
        os.system(f'cp -r {dset_dir}/labels/{key} {out_dir}/labels/{key}')

def main(   kitti360_root = "data/KITTI-360/", # make sure `ln -s [KITTI360-root] ./data/`
            dset_dir = "output/kitti360_v1_512",
            resolution = (512,)*2,):

    ######## Step 1 ########
    print("Step 1: generate data")
    processor = DatasetProcessor(kitti360_root)
    # # generate data: (images/ and labels/)
    processor.process_all_sequences(outdir=dset_dir, semantics_resolution=(512,512))
    #processor.process_all_sequences(outdir="output/kitti360_v1_256/", semantics_resolution=(256,256))

    ######## Step 2 ########
    # generate filters for each sequence, and make visualization videos
    print("Step 2: generate filters for each sequence, and make visualization videos")
    for sequence in kitti360_sequences:
        print("sequence", sequence)
        filters, _ = kitti360_dataset.seq_export_filters(dset_dir=dset_dir, sequence=sequence)
        for filter_name in filters:
            filters[filter_name] = filters[filter_name]
        kept_names = ["full", "basic_filter", "std_filter", "strict_filter"]
        kept_filters = {k:filters[k] for k in kept_names}
        export_datavis_videos(total_trajs=kept_filters["full"].sum(), fps=120, dset_dir=dset_dir, filters=kept_filters, sequence=sequence)

    ######## Step 3 ########
    # combine all sequence's frames together
    print("Step 3: combine all sequence's frames together")
    kitti360_dataset.combine_all_sequence_filters(dset_dir=dset_dir)

    ######## Step 4 ########
    # filter dataset with std_filter, 
    print("Step 4: filter dataset with std_filter")
    filter_dataset(dset_dir=f"{dset_dir}", out_dir=f"{dset_dir}/std_filtered_dset/", filter_name="std_filter")

if __name__ == "__main__":
    main(   kitti360_root = "data/KITTI-360/", # make sure `ln -s [KITTI360-root] ./data/`
            dset_dir = "output/kitti360_v1_512_test",
            resolution = (512,)*2,)
