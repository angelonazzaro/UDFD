<p align="center">
  <img src="./logo.png" alt="Logo" width="200"/>
</p>

# UDFD (Unisa Deep Fake Detector)
UDFD is a deep learning-based tool designed to detect deep fake faces from images. 
The scope of the project is to refine a model that can accurately identify deep fake faces, 
with a focus on improving software engineering aspects like:
 - Model Explainability via Grad-CAM
 - Model Fairness via FairSMOTE balancing of the dataset
 - Model Security creating a module to detect poisonus data attacks
 - ML Ops with a focus on developing an automated pipeline for model retraining and deployment

## Features
The model will usable through a web app developed with Flask with the following features:
- Upload an image to detect deep fake faces
- View the detection results
- Visualize model explainability using Grad-CAM
- Review model output and add training data