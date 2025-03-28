# BMFAN
The codes for our paper "Enhanced Finger Vein Recognition via Bi-Branch Multi-Level Feature Aggregation Network"

The repository includes all the scripts, configurations, and detailed instructions on how to execute the code and reproduce our results. Once the paper is accepted, we will open-source our code.

## Requirements
- Python 3.10.8
- PyTorch 2.1.2+cu118
- NumPy 1.26.4

## Download data
The experimental evaluation utilizes two benchmark biometric datasets:

SDUMLA Dataset: This multimodal biometric database contains 106 subjects' iris, fingerprint, face and gait modalities captured under constrained laboratory conditions.

```
https://time.sdu.edu.cn/kycg/gksjk.htm
```

HKPU Database: The Hong Kong Polytechnic University's public dataset provides 7,920 near-infrared palm images from 400 volunteers. 

```
https://www4.comp.polyu.edu.hk/~csajaykr/fn2.htm
```



## Data pre-processing
   The images of SDUMLA and HKPU can be used directly for training. However, in order to get better recognition results, we pre-processed the images before training:

   1、Pre-process the images using the methods in `/pre-process folder`

   2、Save the processed image as the dataset used in this article.

## Training
```
python train.py 
```

## Evaluate trained models
```
python predict.py 
```
