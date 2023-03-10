# Distracted Driver Detection

Driver distraction constitutes an important factor in the increased risk of road accidents in Bangladesh. These distraction sources can be internal or external.​ In this project, we will detect drivers distracted by internal sources while driving using dash-cam footage.
## Overview
In this project, we have trained CNN architecture like AlexNet and VGG to classify Distracted Driver Detection Dataset from [Kaggle](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data). The classes are:

c0: safe driving​

c1: texting – right​

c2: talking on the phone - right ​

c3: texting – left​

c4: talking on the phone - left ​

c5: operating the radio ​

c6: drinking ​

c7: reaching behind​

c8: hair and makeup ​

c9: talking to passenger

![Performance of the models](https://github.com/shanjidhasan/distracted-driver-detection/blob/master/stats.png)

## Clone GitHub repo
```bash
git clone https://github.com/shanjidhasan/distracted-driver-detection.git
```
## Web Application
### Change branch
```bash
git checkout main
```
### Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```
### Run web app
```bash
streamlit run .\app.py
```

## Train and test models
### Change branch
```bash
git checkout master
```
Download the dataset from this [link](https://drive.google.com/file/d/1oxudncgnPjHl1e0D6-48EIsvl6d8ul19/view?usp=share_link)

Extract the zip in the project folder.
### Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```
### Train models
```bash
python .\train.py -model_name MODEL_NAME -train_dataset_path TRAIN_DATASET_PATH -validation_dataset_path VALIDATION_DATASET_PATH -learning_rate
                LEARNING_RATE -number_of_epoch NUMBER_OF_EPOCH -optimizer OPTIMIZER
```
### Test models
```bash
python .\test.py [-h] -model_name MODEL_NAME -checkpoint_path CHECKPOINT_PATH -test_dataset_path TEST_DATASET_PATH
```
### Project Organization


    ├── README.md          <- The top-level README for developers using this project.
    ├── ddd
    │   ├── test           <- images for testing models
    │   ├── train          <- images for training models
    │   └── val            <- images for validating models
    │
    ├── models
    │   ├── training       <- training data for models
    │   ├── alexnet.pth    <- trained alexnet model
    │   ├── model-drive    <- trained resnet50 model from Kaggle(https://www.kaggle.com/code/longvan92/using-resnet-99-success-rate-on-validation-test)
    │   ├── vgg16.pth      <- trained VGG16 model
    │   └── vgg19.pth      <- trained VGG19 model
    │
    ├── moderArcs          <- Architecture of AlexNet and VGG
    │
    ├── samples            <- Sample images to test the models
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── app.py
    │
    ├── classifier.py
    │
    ├── CSE_3200_1803172.pptx
    │
    ├── labels.txt
    │
    ├── Loader.py
    │
    ├── plotStat.py
    │
    ├── test.py
    │
    └── train.py



## License

[MIT](https://choosealicense.com/licenses/mit/)
