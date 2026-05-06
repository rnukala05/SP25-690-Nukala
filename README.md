# SmartConfuse: Detecting Student Confusion in Online Learning Using Deep Learning

**Name:** Rishwanth Nukala

---

## Project Description

This project focuses on detecting student confusion during online learning sessions using deep learning techniques. The system analyzes short sequences of video frames to identify behavioral patterns such as facial expressions and subtle motion changes that may indicate confusion.

Two models are implemented and compared: a Convolutional Neural Network (CNN) baseline and a CNN combined with a Transformer for sequence modeling. The goal is to evaluate whether temporal modeling improves confusion detection performance.

---

## Setup Instructions

1. Make sure Python 3.8 or above is installed
2. Open terminal or command prompt inside the project folder
3. Install required dependencies using the command below

```
pip install -r requirements.txt
```

---

## Dependencies

The project uses the following main libraries:

* torch
* torchvision
* numpy
* matplotlib
* scikit-learn

All dependencies are listed in the `requirements.txt` file.

---

## How to Run

Run the complete pipeline using the following command:

```
python main.py
```

This will:

* Train the CNN baseline model
* Train the CNN + Transformer model
* Evaluate both models
* Generate all output visualizations

---

## Output Files

After execution, the following files will be created inside the `outputs/` folder:

* confusion_cnn.png
* confusion_transformer.png
* cnn_accuracy.png
* transformer_accuracy.png
* model_comparison.png

These files are used for analysis and reporting.

---

## Project Structure

```
smartconfuse/
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── cnn.py
│   └── transformer.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate.py
│
├── utils/
│   └── utils.py
│
├── outputs/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
```

---

## Summary

This project demonstrates how deep learning models can be used to detect student confusion using visual behavioral data. The comparison between CNN and Transformer-based models helps understand the importance of temporal information in sequence-based tasks.
