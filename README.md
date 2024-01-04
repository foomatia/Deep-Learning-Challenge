# Deep-Learning-Challenge - Alphabet Soup Charity Deep Learning Model Report

## Overview of the Analysis

The purpose of this analysis is to create a binary classification model using machine learning and neural networks to predict the success of organizations funded by Alphabet Soup. The dataset contains information on over 34,000 organizations that have received funding, and the goal is to develop a model that can accurately classify whether an organization will be successful based on various features.

## Results

### Data Preprocessing

1. **Target and Feature Variables:**
   - Target Variable: `IS_SUCCESSFUL`
   - Feature Variables: All columns except `EIN` and `NAME`

2. **Columns Removed:**
   - `EIN` and `NAME` columns were dropped as they are identification columns and are neither targets nor features.

3. **Number of Unique Values:**
   - Determined the number of unique values for each column in the dataset.

4. **Handling Categorical Variables:**
   - Binned "rare" categorical variables together into a new value, 'Other', for columns with more than 10 unique values.

5. **Encoding Categorical Variables:**
   - Used `pd.get_dummies()` to encode categorical variables.

6. **Train-Test Split:**
   - Split the preprocessed data into training and testing datasets using `train_test_split`.

7. **Feature Scaling:**
   - Scaled the training and testing features datasets using `StandardScaler`.

### Compiling, Training, and Evaluating the Model

1. **Neural Network Model:**
   - Created a neural network model with appropriate input features and nodes for each layer using TensorFlow and Keras.
   - Added hidden layers with ReLU activation functions.
   - Utilized a sigmoid activation function for the output layer.
     
![Screenshot 2024-01-04 at 18 58 17](https://github.com/foomatia/Deep-Learning-Challenge/assets/108195931/09928d65-1fbf-4b9c-bdbf-09176cf11e64)

2. **Model Structure:**
   - Checked the structure of the model using the `summary()` method.

![Screenshot 2024-01-04 at 18 59 18](https://github.com/foomatia/Deep-Learning-Challenge/assets/108195931/305c1cea-23c2-47fc-bad2-3fac44888fdf)

3. **Compilation and Training:**
   - Compiled and trained the model with appropriate loss functions and optimizers.

4. **Model Evaluation:**
   - Evaluated the model using the test data to determine the loss and accuracy.

5. **Callback for Saving Weights:**
   - Created a callback that saves the model's weights every five epochs.

6. **Export Results:**
   - Saved and exported the results to an HDF5 file named "AlphabetSoupCharity.h5".

### Optimization of the Model

1. **Model Optimization Methods:**
   - Explored different optimization methods, including adjusting input data, modifying the number of neurons and layers, using different activation functions, and tuning the number of epochs.

2. **New Google Colab File:**
   - Created a new Google Colab file named "AlphabetSoupCharity_Optimization.ipynb".

3. **Preprocessing in Optimization:**
   - Repeated the preprocessing steps in the new Jupyter notebook, adjusting for any modifications made during optimization.

4. **New Neural Network Model:**
   - Designed a new neural network model, implementing at least three model optimization methods.

5. **Export Optimization Results:**
   - Saved and exported the optimized results to an HDF5 file named "AlphabetSoupCharity_Optimization.h5".

## Summary

In summary, the deep learning model was successfully developed to predict the success of organizations funded by Alphabet Soup. The model underwent optimization to achieve a target predictive accuracy higher than 75%. 
Various adjustments were made to the preprocessing steps, the neural network architecture, and other hyperparameters to enhance the model's performance.

### Recommendation for a Different Model

While the neural network model provided satisfactory results, an alternative approach could involve exploring ensemble methods such as Random Forest or Gradient Boosting. Ensemble methods combine multiple models to improve overall predictive performance and can be effective in handling complex relationships in the data. This could serve as an alternative solution, providing a robust and interpretable classification model for the given problem.

Overall, the deep learning model offers a powerful predictive tool, but exploring different models allows for a comprehensive understanding of the data and may uncover alternative solutions to the classification problem.
