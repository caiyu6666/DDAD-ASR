## Guidance for preparing Medical Anomaly Detection Benchmarks

This is the guidance for preprocessing and organizing medical anomaly detection benchmarks: 1) [RSNA Pneumonia Detection Challenge dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), 2) [VinBigData Chest X-ray Abnormalities Detection dataset](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection), 3) [Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), and 4) [LAG dataset](https://ieeexplore.ieee.org/document/8756196).

After preprocessing, all the benchmarks should have the same file structure as the following:


```python
├─DATA_PATH/
│ ├─RSNA/   # data root of one of the 4 datasets
│ │ ├─images/   # preprocessed images of the dataset 
│ │ │ ├─image1.png (or .jpg)
│ │ │ ├─ ......
│ │ ├─data.json   # repartition file of the coressponding dataset
│ ├─VinCXR/
│ │ ├─images/
│ │ │ ├─image1.png (or .jpg)
│ │ │ ├─ ......
│ │ ├─data.json 
...
```

The `data.json` is a dictionary that storing the data repartition information. All our `data.json` files have the same structure as the following: 

```python
{
  "train": {
    "0": ["*.png", ], # The known normal images for one-class training
    "unlabeled": {
          "0": ["*.png", ], # normal images used to build the unlabeled dataset
    	  "1": ["*.png", ]  # abnormal images used to build the unlabeled dataset
    }
  },
  
  "test": {
  	"0": ["*.png", ],  # normal testing images
  	"1": ["*.png", ]  # abnormal testing images
  }
}
```
### RSNA dataset and VinBigData dataset

1. Download the datasets from these links: [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), [VinBigData](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection).
2. Excute `./data/preprocess_cxr.py` to preprocess the two datasets, respectively. (As labels of testing set of the challenge are not available, we only utilize their training set to build our benchmark). Note that `in_dir` and  `out_dir`  should be modify to your corresponding path. The output files should be `*.png`.
3. Place files as shown in the aforementioned file structure tree.



### Brain Tumor MRI dataset and LAG dataset

1. Access the datasets from these links: [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), [LAG](https://ieeexplore.ieee.org/document/8756196).
2. Move all the original training and testing images in one directory `images/`, and place other files as shown in the aforementioned file structure tree.



After organization, all these benchmarks have the same file structure and can be loaded for subsequent operation using the same code. 