# PVCNet

A Deep Learning-based Partial Volume Effect correction method, for Single Photon Emission Computed Tomography (SPECT). 


If you use this in your research, please cite:

	Kaprelian,...


 ## Training Dataset
 
 
 To generate a synthetic dataset of Lu177 sinograms, refer to [this repository](https://github.com/TheoKaprel/PVE_data)
 
 
 The generated dataset should be in .h5 format. Each sample should be a five letter keys (like "ABCDE"). For each sample the training dataset should contain the following keys: 

#### Sinogram-based training (PVCNet-sino)


* "PVfree_att" key containing the ground truth sinogram without PVE (no collimator-detector modeled) obtained with attenuated forward-projection
*  "PVE_att"
*  "PVE_noisy"
*  "rec_fp_att"
*  "attmap_fp"

#### Image-based training (PVCNet-img / PVCNet-hybrid)

* "rec"
* "attmap_4mm"
* "PVCNet_rec"

## Training

To train a PVCNet, use this command:

	python train.py --json configs/config_file.json --output EXAMPLE_REF --output_folder /output/folder/

Two main types of config files: 

#### Sinogram-based training (PVCNet-sino)

Use a file similar to **configs/config_PVCNet_sino.json**, _i.e._: 

* with a dataset/test_dataset containing the sinogram keys (PVfree_att, PVE_att, PVE_noisy, rec_fp_att, attmap_fp)

Two choices: 
* Either with 2 NN (one for denoising, one for PVC). Set

      "network": "unet_denoiser_pvc",

* Either with 1 NN. Set

      "network": "unet",

#### Image-based training (PVCNet-img / PVCNet-hybrid)

Use a config file similar to **configs/config_PVCNet_img.json**.

Once again two choices: 

* Use only "rec" (the reconstructed image with RM)  and the attenuation map as input, then set

      "with_PVCNet_rec": false

This is the _PVCNet-img_.

* Also use "PVCNet_rec" as input, then set

      "with_PVCNet_rec": true

Then, it is the _PVCNet-hybrid_ because it uses the image reconstructed from the sinogram corrected by a previously trained PVCNet-sino.

## Apply 

#### For PVCNet-sino

     python apply.py --input projections.mha --input_rec_fp projections_rec_fp.mha \\
      --attmap_fp attmap_fp.mha --pth path/to/network.pth\\
      --output output_projections.mha

#### For PVCNet-img

    python apply.py --input rec.mhd --attmap_fp attmap_4mm.mha \\
     --pth path/to/network.pth\\
     --output output_image.mha 

#### For PVCNet-hybrid

    python apply.py --input rec.mhd --input_rec_fp rec_noRM_PVCNet_sino_751113.mhd\\
     --attmap_fp attmap_4mm.mha --pth path/to/network.pth\\
     --output output_image.mha 


#### Trained models

Examples of trained models can be found in the "pth" folder
