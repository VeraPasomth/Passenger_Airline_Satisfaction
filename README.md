# Introduction 
The Goal of this project is to determine which independent variables have the most influence on the 
dependent variable, which is in this case the satisfaction of the passengers. It is important for an airline 
to know what customers value the most, in order to improve the customer experience and increase the 
customer flow.

The scipt Data_Exploration can be run to preprocess and analyze the dataset. The preprocessing steps applied are as follows:

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
