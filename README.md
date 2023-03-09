# Breast Cancer Detection in Mammograms Using Deep Learning Techniques [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3985051.svg)](https://doi.org/10.5281/zenodo.3985051) [![GitHub license](https://img.shields.io/github/license/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning)](https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning/blob/master/LICENSE)

## Updates

I have been working since the end of my Master's in 2020 to publish this dissertation in a renown journal. Here are the latest updates of this project:

### 2023

* The paper, entitled "*A divide and conquer approach to maximise deep learning mammography classification accuracies*", was formally accepted for publication in January 2023.
* The paper is in the publication pipeline and will soon be published.

### 2022

* Due to the time that has passed since the initial dissertation project was finished, help was enlisted by including my supervisor's PhD student. The goal was to:
  * update the literature review with more recent papers,
  * reproduce the results and update the code/instructions for installation and setup (code can be found in this updated repository: [Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication](https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication)),
  * have an extra pair of eyes to polish the whole paper and keep the flow consistent.
* The paper was submitted to PLOS ONE in July 2022.
* Reviewers got back to us in September 2022 with amendments for the paper to be considered for publication. These amendments included revamping the narrative to highlight the main contribution of the paper, add additional sections underlining the processes and decision-making process, and general amendments for the paper's flow and style to remain consistent throughout.
* The amended version was sent in October.

### 2021

* Elected a journal to aim to publish in: PLOS ONE.
* Began transforming the project from a dissertation report to a paper (choice of language, trimming down, changing narrative, updating figures).
* After multiple iterations and feedback from both my supervisors, a first draft was created.

### 2020

* Completed my Master's degree and submitted my dissertation. Final results: 90%.
* Began discussions with my supervisors to consider publishing the dissertation in a renown journal.

## What can I find in this repository?

You can find the full dissertation project (code + report) for the MSc Artificial Intelligence at the University of St Andrews (2020).

The final report can be read here: [Breast Cancer Detection in Mammograms using Deep Learning Techniques, Adam Jaamour (2020)](https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning/blob/master/Breast%20Cancer%20Detection%20in%20Mammograms%20using%20Deep%20Learning%20Techniques%20-%20Adam%20Jaamour%2C%202020.pdf)

The publication of this project can be found here:
* Paper published in PLOS ONE: *[soon]*
* Peer-reviewed code: https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication

## Abstract

The objective of this dissertation is to explore various deep learning techniques that can be used to implement a system which learns how to detect instances of breast cancer in mammograms. Nowadays, breast cancer claims 11,400 lives on average every year in the UK, making it one of the deadliest diseases. Mammography is the gold standard for detecting early signs of breast cancer, which can help cure the disease during its early stages. However, incorrect mammography diagnoses are common and may harm patients through unnecessary treatments and operations (or a lack of treatments). Therefore, systems that can learn to detect breast cancer on their own could help reduce the number of incorrect interpretations and missed cases.

Convolution Neural Networks (CNNs) are used as part of a deep learning pipeline initially developed in a group and further extended individually. A bag-of-tricks approach is followed to analyse the effects on performance and efficiency using diverse deep learning techniques such as different architectures (VGG19, ResNet50, InceptionV3, DenseNet121, MobileNetV2), class weights, input sizes, amounts of transfer learning, and types of mammograms.

![CNN Model](https://i.postimg.cc/wxWB8CTP/CNN-architecture.png)

Ultimately, 67.08\% accuracy is achieved on the CBIS-DDSM dataset by transfer learning pre-trained ImagetNet weights to a MobileNetV2 architecture and pre-trained weights from a binary version of the mini-MIAS dataset to the fully connected layers of the model. Furthermore, using class weights to fight the problem of imbalanced datasets and splitting CBIS-DDSM samples between masses and calcifications also increases the overall accuracy. Other techniques tested such as data  augmentation and larger image sizes do not  yield increased accuracies, while the mini-MIAS dataset proves to be too small for any meaningful results using deep learning techniques. These results are compared with other papers using the CBIS-DDSM and mini-MIAS datasets, and with the baseline set during the implementation of a deep learning pipeline developed as a group.

## Usage on a GPU lab machine

Clone the repository:

```
cd ~/Projects
git clone https://github.com/Adamouization/Breast-Cancer-Detection-Code
```

Create a repository that will be used to install Tensorflow 2 with CUDA 10 for Python and activate the virtual environment for GPU usage:

```
cd libraries/tf2
tar xvzf tensorflow2-cuda-10-1-e5bd53b3b5e6.tar.gz
sh build.sh
```

Activate the virtual environment:

```
source /cs/scratch/<username>/tf2/venv/bin/activate
```

Create `output`and `save_models` directories to store the results:

```
mkdir output
mkdir saved_models
```

`cd` into the `src` directory and run the code:

```
main.py [-h] -d DATASET [-mt MAMMOGRAMTYPE] -m MODEL [-r RUNMODE] [-lr LEARNING_RATE] [-b BATCHSIZE] [-e1 MAX_EPOCH_FROZEN] [-e2 MAX_EPOCH_UNFROZEN] [-roi] [-v] [-n NAME]
```

where:
* `-h` is a flag for help on how to run the code.
* `DATASET` is the dataset to use. Must be either `mini-MIAS`, `mini-MIAS-binary` or `CBIS-DDMS`. Defaults to `CBIS-DDMS`.
* `MAMMOGRAMTYPE` is the type of mammograms to use. Can be either `calc`, `mass` or `all`. Defaults to `all`.
* `MODEL` is the model to use. Must be either `VGG-common`, `VGG`, `ResNet`, `Inception`, `DenseNet`, `MobileNet` or `CNN`.
* `RUNMODE` is the mode to run in (`train` or `test`). Default value is `train`.
* `LEARNING_RATE` is the optimiser's initial learning rate when training the model during the first training phase (frozen layers). Defaults to `0.001`. Must be a positive float.
* `BATCHSIZE` is the batch size to use when training the model. Defaults to `2`. Must be a positive integer.
* `MAX_EPOCH_FROZEN` is the maximum number of epochs in the first training phrase (with frozen layers). Defaults to `100`.
* `MAX_EPOCH_UNFROZEN`is the maximum number of epochs in the second training phrase (with unfrozen layers). Defaults to `50`.
* `-roi` is a flag to use versions of the images cropped around the ROI. Only usable with mini-MIAS dataset. Defaults to `False`.
* `-v` is a flag controlling verbose mode, which prints additional statements for debugging purposes.
* `NAME` is name of the experiment being tested (used for saving plots and model weights). Defaults to an empty string.

## Dataset installation

#### mini-MIAS dataset

* This example will use the [mini-MIAS](http://peipa.essex.ac.uk/info/mias.html) dataset. After cloning the project, travel to the `data/mini-MIAS` directory (there should be 3 files in it).

* Create `images_original` and `images_processed` directories in this directory: 

```
cd data/mini-MIAS/
mkdir images_original
mkdir images_processed
```

* Move to the `images_original` directory and download the raw un-processed images:

```
cd images_original
wget http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz
```

* Unzip the dataset then delete all non-image files:

```
tar xvzf all-mias.tar.gz
rm -rf *.txt 
rm -rf README 
```

* Move back up one level and move to the `images_processed` directory. Create 3 new directories there (`benign_cases`, `malignant_cases` and `normal_cases`):

```
cd ../images_processed
mkdir benign_cases
mkdir malignant_cases
mkdir normal_cases
```

* Now run the python script for processing the dataset and render it usable with Tensorflow and Keras:

```
python3 ../../../src/dataset_processing_scripts/mini-MIAS-initial-pre-processing.py
```

#### DDSM and CBIS-DDSM datasets

These datasets are very large (exceeding 160GB) and more complex than the mini-MIAS dataset to use. They were downloaded by the University of St Andrews School of Computer Science computing officers onto \textit{BigTMP}, a 15TB filesystem that is mounted on the Centos 7 computer lab clients with NVIDIA GPUsusually used for storing large working data sets. Therefore, the download process of these datasets will not be covered in these instructions.\\

The generated CSV files to use these datasets can be found in the `/data/CBIS-DDSM` directory, but the mammograms will have to be downloaded separately. The DDSM dataset can be downloaded [here](http://www.eng.usf.edu/cvprg/Mammography/Database.html), while the CBIS-DDSM dataset can be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#5e40bd1f79d64f04b40cac57ceca9272).

## License 
* see [LICENSE](https://github.com/Adamouization/Breast-Cancer-Detection-and-Segmentation/blob/master/LICENSE) file.

## Code Authors

* Adam Jaamour
* Ashay Patel
* Shuen-Jen Chen

The common pipeline can be found at [DOI 10.5281/zenodo.3975092](https://zenodo.org/record/3975093)

## Contact
* Email: adam@jaamour.com
* Website: www.adam.jaamour.com
* LinkedIn: [linkedin.com/in/adamjaamour](https://www.linkedin.com/in/adamjaamour/)
