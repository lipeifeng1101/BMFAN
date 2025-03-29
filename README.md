# Bi-Branch Multi-Level Feature Aggregation Network (BMFAN)
**Official Implementation of**  
_"Enhanced Finger Vein Recognition via Bi-Branch Multi-Level Feature Aggregation Network"_  
*(Under Review, The visual computer 2025)*



This repository provides the complete implementation of our novel BMFAN framework, which achieves state-of-the-art performance on multiple finger vein recognition benchmarks through three key innovations:

1. A Bi-Branch Multi-Level Feature Aggregation Network (BMFAN) for finger vein recognition is designed, which enhances recognition accuracy and discriminative capability through feature extraction, enhancement, and multi-level aggregation.

2. We propose a Local Feature Refinement Unit (LFRU) and a Global Feature Refinement Unit (GFRU), incorporating attention frameworks to enhance the aggregation of global topology, strengthen the representational power of local details, and ultimately improve recognition accuracy.

3. We design a Multi-Level Feature Aggregation Unit (MLVF-AU), incorporating residual connections and Layer Normalization (LN), which effectively preserves the expressiveness and discriminative power of the features, thereby optimizing the performance of the classification task.

## 📦 Repository Contents
| Directory            | Contribution to Paper Research       |
|----------------------|------------------------------------- |
| `BMFAN/model.py`     | Overall network implementation       |
| `BMFAN/my.dataset.py`| Data processing loading              |
| `BMFAN/predict.py`   | Predict                              |
| `BMFAN/split.py`     | Dataset segmentation                 |
| `BMFAN/train.py`     | Train                                |
| `BMFAN/utils.py`     | Parameter settings                   |
| `BMFAN/pre-process/` | Dataset preprocessing                |


## 🛠️ Environment Setup
- Python 3.10.8
- PyTorch 2.1.2+cu118
- NumPy 1.26.4

## 📥 Dataset Preparation
The experimental evaluation utilizes two benchmark biometric datasets:

SDUMLA Dataset: This multimodal biometric database contains 106 subjects' iris, fingerprint, face and gait modalities captured under constrained laboratory conditions.

```
https://time.sdu.edu.cn/kycg/gksjk.htm
```

HKPU Database: The Hong Kong Polytechnic University's public dataset provides 7,920 near-infrared palm images from 400 volunteers. 

```
https://www4.comp.polyu.edu.hk/~csajaykr/fn2.htm
```



## 🧩 Data pre-processing
   The images of SDUMLA and HKPU can be used directly for training. However, in order to get better recognition results, we pre-processed the images before training:

   1、Pre-process the images using the methods in `BMFAN/pre-process`

   2、Save the processed image as the dataset used in this article.

## 🚀 Model Training
```
python train.py 
```

Key training parameters from paper:

Learning rate: 0.0009

Batch size: 32 (per GPU)

Epoch: 400

## 📊 Evaluation
```
python predict.py 
```

Evaluation metrics implemented:

Accuracy (ACC)

Equal Error Rate (EER)
