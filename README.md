# **Weed Species Classification and Bounding Box Regression**
Leveraging advanced image processing and deep learning, this project focuses on CNNs and the Keras API for image processing and regression tasks related to plant images, particularly weed species from [Plant Seedlings dataset](https://vision.eng.au.dk/plant-seedlings-dataset/)"I worked on a subset". The project involves data preparation, basic transfer learning using the VGG-16 model, classification, and regression networks. Regularization methods are applied to improve the model, and discussions on overfitting and the impact of regularization are included. The submission requires a Jupyter file containing the solution, and late submissions are not allowed. It's contributes to understanding CNNs, transfer learning, and handling small training data. This project holds significance within my Master's in Computer Vision at uOttawa (2023).
<p align="center">
   <img src="https://github.com/RimTouny/Weed-Species-Classification-and-Bounding-Box-Regression/assets/48333870/cf5cd642-c2a6-49a8-9b82-d33b2098aa4a"/>
</p>	

- Required libraries: scikit-learn, pandas, matplotlib.
- Execute cells in a Jupyter Notebook environment.
- The uploaded code has been executed and tested successfully within the [Google Colab](https://colab.google/) environment.

## Image classification and bounding box regression using transfer learning with a VGG-16 model.
The dataset comprises 4 classes with 250 images each, divided into training,and testing sets, images size are differnet: Cleavers, Common Chickweed, Maize, Shepherd’s Purse,

## **Key Tasks Undertaken**    
1. **Data Preparation:**
   - Uploaded a subset of the dataset  from Google Drive.
   - Extracted the dataset and organized it into 70% training, 15% validation, and 15% testing sets.
      + Traning Set
        <p align="center">
          <img src="https://github.com/RimTouny/Weed-Species-Classification-and-Bounding-Box-Regression/assets/48333870/3dbbb310-7c09-4e7c-8097-9a0c14b28f46"/>
      </p>	

      + Validation Set
        <p align="center">
          <img src="https://github.com/RimTouny/Weed-Species-Classification-and-Bounding-Box-Regression/assets/48333870/d38e9c50-e1c0-46f9-adb1-4d733fe210af"/>
      </p>	

      + Testing Set
        <p align="center">
          <img src="https://github.com/RimTouny/Weed-Species-Classification-and-Bounding-Box-Regression/assets/48333870/44c4ceab-f0b6-437a-922c-bd1c8f035df1"/>
      </p>	


   - Loaded the data, resized images to 32x32 pixels, and created DataFrames for each set.
   ```python
      Training Data Size: 700
      Training Data Label Counts:
      Shepherds_Purse     175
      Common_Chickweed    175
      Cleavers            175
      Maize               175
      Name: Label, dtype: int64 
      
      Size of the Images in Training Data: (32, 32, 3)
      ----------------------------------------------------------------
      
      Validation Data Size: 148
      Validation Data Label Counts:
      Shepherds_Purse     37
      Common_Chickweed    37
      Cleavers            37
      Maize               37
      Name: Label, dtype: int64 
      
      Size of the Images in Validation Data: (32, 32, 3)
      ----------------------------------------------------------------
      
      Test Data Size: 152
      Test Data Label Counts:
      Shepherds_Purse     38
      Common_Chickweed    38
      Cleavers            38
      Maize               38
      Name: Label, dtype: int64
      Size of the Images in Test Data: (32, 32, 3)
      ----------------------------------------------------------------
      ```

2. **Classification Network (Transfer Learning):**
   - Used the VGG-16 model for transfer learning.
   - Modified the model by adding custom layers for classification.
   ```python
      # Add custom layers
      x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
      x = MaxPooling2D((2, 2))(x)
      x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
      x = MaxPooling2D((2, 2))(x)
      x = Flatten()(x)
      outputs = Dense(4, activation='softmax')(x)  # Output layer for 4 classes
      ```
   - One-hot encoded the labels.
   - Trained the classification model, monitored convergence, and visualized learning curves.
      <p align="center">
         <img src="https://github.com/RimTouny/Weed-Species-Classification-and-Bounding-Box-Regression/assets/48333870/63b7cf41-037e-455c-b874-ec41e39062e0"/>
      </p>	

   - Plotted and analyzed the confusion matrix for training, validation, and testing datasets.
      <p align="center">
         <img src="https://github.com/RimTouny/Weed-Species-Classification-and-Bounding-Box-Regression/assets/48333870/086a16b5-9653-46cd-8f30-4298cf2bcf47"/>
      </p>

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
