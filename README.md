# Predicting-Disease-Spread

## Objective
Dengue fever is a mosquito-borne disease that occurs in tropical and sub-tropical parts of the world. In mild cases, symptoms are similar to the flu: fever, rash, and muscle and joint pain. In severe cases, dengue fever can cause severe bleeding, low blood pressure, and even death. Using environmental data collected by various U.S. Federal Government agencies—from the Centers for Disease Control and Prevention to the National Oceanic and Atmospheric Administration in the U.S. Department of Commerce, this project is trying to predict the number of dengue fever cases reported each week in San Juan, Puerto Rico and Iquitos, Peru.

![alt text](https://github.com/ys3197/Predicting-Disease-Spread/blob/main/Images/project_preview.PNG)


## Data
The data has been obtained from *drivenData.org*, it was released as a part of their competition *DengAI*. I also included the training data in the [Data](https://github.com/ys3197/Predicting-Disease-Spread/tree/main/Data/RawData) folder.


## Project Structure

- **Explore Data Analysis**
  - EDA is always important in the machine learning project therefore we obtain a individual notebook for exploring the data.
 
- **Feature Engineer**
  - Extract useful features and generate new features
 
- **Models**
  - One Model or Two Model by locations
  - Hyperparameter Tuning by Optuna
  - Feature Importance Analysis by Shap
