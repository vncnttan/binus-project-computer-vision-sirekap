# SIREKAP
Created for AOL BINUS Computer Vision Project

### Background
The current vote tallying system, SiRekap, has faced many criticisms for being inaccurate and unreliable. SiRekap often has OCR errors and logical flaws causing people to lose trust in the transparency of the election process. Knowing these concerns, we initiated BiRekap where we aim to make a more reliable and accurate ballot counting using advanced AI and Computer Vision. Through BiRekap, we hope we can rebuild people’s trust in the system and transparency of the election. 


### Dataset Overview
The dataset consists of images of ballot sheet from each TPS taken from 2024 president election and a csv file of ballot count for each candidate. Dataset is taken from Data Science Competition - Gammafest IPB 2024 | Kaggle originally purposed for Gammafest Data Science Competition. The dataset consists of: 

`Test/`: 200 ballot images from TPS_501.jpg to TPS_700.jpg, unlabeled 
`Train/`: 500 ballot images from TPS_001.jpg to TPS_500.jpg labelled in label.csv 
`label.csv`: Actual ballot count of each candidate for each TPS on the Train folder 
`sample_submission.csv`: Illustration file for competition submission. 

For our purposes, we created manual_ans.csv for the label of test images with the same format of label.csv for ballot images in the test folder (from TPS_501.jpg to TPS_700.jpg). This label will be purposeful for evaluating loss and accuracy from the test images. The train dataset will be further split into train set and validation set. 

### File Description
