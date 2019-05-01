# Distracted Driver MultiAction Classification

**Data set** : https://www.kaggle.com/c/state-farm-distracted-driver-detection

**Objective** : Classify images into these 10 classes 
  - c0: safe driving
  - c1: texting - right
  - c2: talking on the phone - right
  - c3: texting - left
  - c4: talking on the phone - left
  - c5: operating the radio
  - c6: drinking
  - c7: reaching behind
  - c8: hair and makeup
  - c9: talking to passengerType some Markdown on the left

**Data Preparation** 

*Code* : Distracted_Driver_MultiAction_Classification.ipynb

1. Iterating and Reading the Training and Test files
2. Scaling images to 240x240

**Model Building** 

*Code* : Distracted_Driver_MultiAction_Classification.ipynb

Implimented done in TensorFlow 2.0 using CNN architecture 

1. Inastall Tensorflow 2.0 and use tf.keras as high level api
2. Model has 3 CNN layers (each layer has Conv2D+BatchNormalization+Conv2D+BatchNormalization+MaxPooling2D+Dropout), Flattening layer and Dense Layers
3. Model Compilatation with loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam'
4. Early Stopping with patience of 5
5. Model Fit with Batch of 50 and 15 Epochs
6. Save the Model

**Model Building** 

*Code* : Distracted Driver Load Model.ipynb

1. Read Test Images
2. Scaling images to 240x240
3. Load the Saved model
4. Predict or validate
