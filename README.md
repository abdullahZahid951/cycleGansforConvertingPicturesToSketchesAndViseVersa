# CycleGAN: Face to Sketch and Sketch to Face Image-to-Image Translation

This project implements a **CycleGAN** model to perform image-to-image translation between face images and sketches. Using the provided **Person Face Sketches** dataset, the model is trained to convert:
- A real face image into a corresponding sketch.
- A sketch into a corresponding real face image.

The model is trained end-to-end, following the CycleGAN framework described in the original research paper.

---
## Table of Contents
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [User Interface](#user-interface)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## Objectives
1. **Image-to-Image Translation**:
    - Convert a real face image into a sketch.
    - Convert a sketch into a real face image.
2. **Training Features**:
    - Save model weights after every epoch to resume training if interrupted.
    - Implement end-to-end training with CycleGAN using the **Person Face Sketches** dataset.
3. **Deployment**:
    - Create a user-friendly web interface for real-time image conversion using Streamlit.

---
## Dataset
- The **Person Face Sketches Dataset**  (https://www.kaggle.com/datasets/almightyj/person-face-sketches)  is used, which contains paired data of face images and their corresponding sketches.
- Directory Structure:
- /photosandsketches/ train/ photos/  ,   sketches/ test/ photos/ sketches/
- Images are resized to **64x64** pixels for training due to memory constraints.

---

## Model Architecture
The project uses two main components:
1. **Generator Networks**:
  - Converts real face images to sketches and vice versa.
  - Includes 9 residual blocks for feature transformation.

2. **Discriminator Networks**:
  - Uses PatchGAN to classify real and generated images.
  - Operates on image patches to reduce computational load.

### Loss Functions
- **GAN Loss**: Encourages the generators to produce realistic outputs.
- **Cycle Consistency Loss**: Ensures input-output mapping consistency.
- **Optimization**: Adam optimizer with a learning rate of `0.0002` and `betas=(0.5, 0.999)`.

---

---

## Model Architecture
The project uses two main components:
1. **Generator Networks**:
  - Converts real face images to sketches and vice versa.
  - Includes 9 residual blocks for feature transformation.

2. **Discriminator Networks**:
  - Uses PatchGAN to classify real and generated images.
  - Operates on image patches to reduce computational load.

### Loss Functions
- **GAN Loss**: Encourages the generators to produce realistic outputs.
- **Cycle Consistency Loss**: Ensures input-output mapping consistency.
- **Optimization**: Adam optimizer with a learning rate of `0.0002` and `betas=(0.5, 0.999)`.

---


## Training Process
1. **Hyperparameters**:
  - Batch size: `256`
  - Epochs: `15`
  - Cycle consistency weight: `λ=10`
2. **Weight Initialization**:
  - All convolutional layers are initialized using a normal distribution (`mean=0, std=0.02`).
3. **Pretrained Weights**:
  - The model resumes from saved weights (`.pth` files) if available, or starts from scratch.

### Training Loop
- Each epoch involves:
- Training the **Discriminators**: Distinguish real and generated images.
- Training the **Generators**: Minimize GAN and cycle consistency losses.
- After each epoch:
- Save model weights.
- Log training progress and losses.

---
## User Interface
The project includes a Flask-based web application to interact with the trained model.

### Features
1. Upload an image or use live camera input.
2. Perform the following operations:
 - Convert a **face image** to a **sketch**.
 - Convert a **sketch** to a **face image**.
3. Real-time results displayed in the browser.

### How to Run
```bash
# Clone the repository
git clone https://github.com/abdullahZahid951/cycleGansforConvertingPicturesToSketchesAndViseVersa.git

# If you have compute (training for more epochs)
run cyclegansTraining.ipynb
# make sure to put your weights in appForCycleGans.py

#Wanted To run the UI
streamlit run appForCycleGans.py  
 


