# Microplastic Detection in Water Using Deep Convolutional Neural Networks

--------------------------------------------------------------------------------

SETTING UP THE PROJECT
-------

Before starting the inference paste folders from drive: https://drive.google.com/drive/folders/1RaGRy-zwqtYEE_VeE_hTn8V12x76jxhN?usp=sharing

dataset_cropped_224x224_newest.zip paste to the folder ./content/dataset_cropped_224x224_newest/
models.zip paste to the folder ./

--------------------------------------------------------------------------------

DATASET
-------

The dataset is provided in: ./content/dataset_cropped_224x224_newest/dataset_cropped_224x224_newest.zip

All images are 224x224 pixels.

Before running any inference script, **unzip the dataset**.
    unzip content/dataset_cropped_224x224_newest/dataset_cropped_224x224_newest.zip -d content/dataset_cropped_224x224_newest/

--------------------------------------------------------------------------------

MODELS
------

All trained model weights and their training history files are located in: ./models/

Each subfolder corresponds to a specific model or experiment
The `.pth` files are loaded by the inference scripts.

--------------------------------------------------------------------------------

INFERENCE SCRIPTS
-----------------

All runnable inference pipelines are located in:
    ./inference_scripts/

Each script:
- loads the required model,
- loads its `.pth` weights from ./models,
- loads the validation dataset,
- performs inference,
- outputs predictions, metrics

Run any inference script with:
    python inference_scripts/<script_name>.py

--------------------------------------------------------------------------------

NOTEBOOKS
---------

The notebooks in:
    ./notebooks/

contain the original training and validation code that produced the models.  
They are **for reference only** and not required to run inference.

--------------------------------------------------------------------------------

INSTALLATION
------------

Install all required packages with:
    pip install -r requirements.txt
Make sure your PyTorch version matches your CUDA setup if you use GPU acceleration.

--------------------------------------------------------------------------------

FULL WORKFLOW
-------------

1. Unzip the dataset:
    unzip content/dataset_cropped_224x224_newest/dataset_cropped_224x224_newest.zip -d content/dataset_cropped_224x224_newest/

2. Install dependencies:
    pip install -r requirements.txt

3. Run an inference script:
    python inference_scripts/<script_name>.py

--------------------------------------------------------------------------------

NOTES
-----

- Do not modify the dataset folder structure; inference scripts expect it unchanged.
- All inference depends on `.pth` model files stored inside ./models.
- Training notebooks document the development workflow but do not need to be run.
- All inference scripts count the number of params and MACs ONLY for the backbone model, excluding F-RCNN, RPN and FPN params and MACs

--------------------------------------------------------------------------------
