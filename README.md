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
   The image features of Flickr30K and MS-COCO are available in numpy array format, which can be used for training directly. However, if you wish to test on another dataset, you will need to start from scratch:

   
   1、Use the and the bottom-up attention model to extract features of image regions. The output file format will be a tsv, where the columns are ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features'].  `bottom-up-attention/tools/generate_tsv.py`

   
   2、Use to convert the above output to a numpy `array.util/convert_data.py`
