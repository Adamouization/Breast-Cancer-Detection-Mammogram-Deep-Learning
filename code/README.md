# Breast-Cancer-Detection-Code

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
source /Breast-Cancer-Detection-Code/tf2/venv/bin/activate
```

`cd` into the `src` directory and run the code:

```
python main.py [-h] -d DATASET -m MODEL [-r RUNMODE] [-i IMAGESIZE] [-v]
```

where:
* `-h` is a  flag for help on how to run the code.
* `DATASET` is the dataset to use. Must be either `mini-MIAS` or `CBIS-DDMS`.
* `MODEL` is the model to use. Must be either `basic` or `advanced`.
* `RUNMODE` is the mode to run in (`train` or `test`). Default value is `train`.
* `IMAGESIZE` is the image size to feed into the CNN model (`small` - 512x512px; or `large` - 2048x2048px). Default value is `small`.
* `-v` is a flag controlling verbose mode, which prints additional statements for debugging purposes.

## Dataset usage

### mini-MIAS dataset

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
python3 ../../../src/data_manipulations/mini-MIAS-initial-pre-processing.py
```

### DDSM and CBIS-DDSM datasets

These datasets are very large (exceeding 160GB) and more complex than the mini-MIAS dataset to use. Downloading and pre-processing them will therefore not be covered by this README. 

Our generated CSV files to use these datasets can be found in the `/data/CBIS-DDSM` directory, but the mammograms will have to be downloaded separately. The DDSM dataset can be downloaded [here](http://www.eng.usf.edu/cvprg/Mammography/Database.html), while the CBIS-DDSM dataset can be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#5e40bd1f79d64f04b40cac57ceca9272).
