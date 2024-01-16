# **Weed Species Classification and Bounding Box Regression**
Leveraging advanced image processing and deep learning, this project focuses on CNNs and the Keras API for image processing and regression tasks related to plant images, particularly weed species from [Plant Seedlings dataset](https://vision.eng.au.dk/plant-seedlings-dataset/)"I worked on a subset `a3 dataset.zip`". The project involves data preparation, basic transfer learning using the VGG-16 model, classification, and regression networks. Regularization methods are applied to improve the model, and discussions on overfitting and the impact of regularization are included. The submission requires a Jupyter file containing the solution, and late submissions are not allowed. It's contributes to understanding CNNs, transfer learning, and handling small training data. This project holds significance within my Master's in Computer Vision at uOttawa (2023).

- Required libraries: scikit-learn, pandas, matplotlib.
- Execute cells in a Jupyter Notebook environment.
- The uploaded code has been executed and tested successfully within the [Google Colab](https://colab.google/) environment.

## Image classification and bounding box regression using transfer learning with a VGG-16 model.
The dataset comprises 4 classes with 250 images each, divided into training,and testing sets, images size are differnet: Cleavers, Common Chickweed, Maize, Shepherd’s Purse,

## **Key Tasks Undertaken**    
1. **Data Preparation:**
   - Uploaded a dataset (`a3_dataset.zip`) from Google Drive.
   - Extracted the dataset and organized it into training, validation, and testing sets.
   - Loaded the data, resized images to 32x32 pixels, and created DataFrames for each set.

2. **Classification Network (Transfer Learning):**
   - Used the VGG-16 model for transfer learning.
   - Modified the model by adding custom layers for classification.
   - One-hot encoded the labels.
   - Trained the classification model, monitored convergence, and visualized learning curves.
   - Plotted and analyzed the confusion matrix for training, validation, and testing datasets.

3. **Regression Network (Transfer Learning):**
   - Loaded bounding box dimensions from the `bbox.json` file.
   - Normalized height and width values.
   - Split the data into training, validation, and testing sets.
   - Used VGG-16 for transfer learning with custom layers for regression.
   - Trained the regression model, monitored convergence, and visualized learning curves.
   - Calculated mean squared error and mean absolute error for training, validation, and testing datasets.

4. **Model Improvement (Classification Network):**
   - Modified the VGG-16 model by adding extra Keras layers.
   - Introduced regularization techniques such as Batch Normalization and Dropout.
   - Trained the improved classification model, monitored convergence, and visualized learning curves.
   - Plotted and analyzed the confusion matrix for training, validation, and testing datasets.

5. **Discussion and Analysis:**
   - Discussed and analyzed the results, including the accuracy, mean squared error, and mean absolute error for both classification and regression tasks.
   - Considered the impact of limited dataset size on model generalization.

6. **Further Improvement:**
   - Suggested potential improvements, such as increasing the dataset size to mitigate overfitting and enhance generalization.

Overall, your code covers a comprehensive machine learning workflow, addressing both classification and regression tasks with detailed analysis and discussions. If you have any specific questions or tasks you'd like assistance with, feel free to let me know!
