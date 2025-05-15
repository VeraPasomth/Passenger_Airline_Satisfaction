# Software Description
The Goal of this project is to determine which independent variables have the most influence on the 
dependent variable, which is in this case the satisfaction of the passengers. It is important for an airline 
to know what customers value the most, in order to improve the customer experience and increase the 
customer flow.
## Data Exploration
The script _Data_Exploration_ can be run to preprocess and analyze the [dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction). This generates a csv file contianing the cleaned data. The preprocessing steps applied are as follows:

**1. Binary Encode Categorical Values and Label Encode Nominal Values:**
Encoding the categorial variables was necessary because all the values need to be in numerical 
form to be processed by the models.

**2. Replace Empty Spaces with ‘_’ in Column Labels:**
Replacing the empty spaces in the column labels made it easier to select. 

**3. Shorten ‘Departure and Arrival Time Convenience’ label:**
The label for Departure and Arrival Time Convenience was shortened to DATC to visualize the 
heatmap more easily. 

**4. Drop Unnecessary Columns:**
The columns dropped were ID, Unnamed: 0. To prevent multicollinearity, Departure Delay was 
also removed since it had high correlation with Arrival Delay. 

**5. Detect and Fill Missing values.**
Filling in missing data was done in order to maintain representation and reduce bias for these 
samples. 

**6. Removal of Outliers**
Outliers from the Arrival Delay were removed because airline delays of more than 500 minutes 
are quite unlikely and could be there by error.

**7. Split dataset in training and testing dataset**
Lastly the dataset is split into 80% training and 20% testing. 

## Machine Learning Models
The script _KNN_Bayes_Model_ starts training and eventually evaluates the models using a confusion matrix and area under the curve charts.

By running the script _RF_NB_Model_, this trains a random forest and naive bayes model and generates learning curves to determine whether the models are over-fitting or underfitting. Once training is completed the models are evaluated using accuracy. The models are then compared using a Wilcoxon Signed Rank test to find if there is a significant difference between the two models.
