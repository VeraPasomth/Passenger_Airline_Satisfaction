# Data Preprocessing
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix,accuracy_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.patches import Ellipse

#Load data from CSV file and Shorten Column Name
df = pd.read_csv("airline_passenger_satisfaction.csv", index_col=False)
df.rename(columns={"Departure and Arrival Time Convenience": "DATC"}, inplace=True)

#remove ID column
df.drop('ID', axis=1, inplace=True)

#replace empty space in column labels
df.columns = [c.replace(' ', '_') for c in df.columns]

columns_for_label_encoding = ["Class", "Customer_Type", "Gender", "Type_of_Travel", "Satisfaction"]
labelencoder = LabelEncoder()
for column in columns_for_label_encoding:
    df[column] = df[column].astype(str)
    df[column] = labelencoder.fit_transform(df[column])

# Arrange column order
dataFrame = pd.DataFrame(df, columns=['Gender', 'Customer_Type', 'Type_of_Travel', 'Class',
                                      'Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay',
                                      'DATC', 'Ease_of_Online_Booking', 'Check-in_Service', 'Online_Boarding',
                                      'Gate_Location', 'On-board_Service', 'Seat_Comfort', 'Leg_Room_Service',
                                      'Cleanliness', 'Food_and_Drink', 'In-flight_Service', 'In-flight_Wifi_Service',
                                      'In-flight_Entertainment', 'Baggage_Handling', 'Satisfaction'])

outlierDf = pd.DataFrame(df, columns=['Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay'])

mean = round(dataFrame['Arrival_Delay'].mean())
dataFrame.loc[(dataFrame.Arrival_Delay > 500),'Arrival_Delay']= mean

#Fill-in Missing Values with Mean
dataFrame['Arrival_Delay'] = dataFrame['Arrival_Delay'].fillna(mean)
#Export to CSV File
dataFrame.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')

#Load Preprocessed Data
df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)
QuantitativeDF = pd.DataFrame(df, columns=['Age','Flight_Distance','Arrival_Delay'])
df = df.drop('Unnamed: 0', axis=1)


#Update Preprocessed CSV File
df = df.drop('Departure_Delay', axis=1)
df.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')

df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)

#Specify Dependent Variable Columns
independent_var = df.filter(items = ["Age", "Flight_Distance", "Arrival_Delay"])
dependent_var = df.iloc[0:, 22:].values.ravel()

def accuracy(name, n_clusters, labels, clusters):

    for i in range(n_clusters):
        cat = (clusters == i)
        labels[cat] = mode(dependent_var[cat])[0]

    acc = accuracy_score(dependent_var, labels)
    print("Accuracy {}: {}".format(name,acc))

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(independent_var)

    print(kmeans)
    wcss.append(kmeans.inertia_)



plt.plot(range(1, 11), wcss)
plt.title('Selecting the Number of Clusters using the Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
#determine number of clusters
print("number of clusters: ",kl.elbow)

k_means_optimum = KMeans(n_clusters = kl.elbow, init='k-means++',  random_state=42)
k_means_output = k_means_optimum.fit_predict(independent_var)

colormap = np.array(['red', 'lime', 'blue'])
plt.suptitle('Kmeans Clusters')
plt.subplot(2, 2, 1)
plt.title("Plot 1")
p1 = plt.scatter(independent_var["Age"], independent_var["Flight_Distance"],c=k_means_output,cmap="viridis", s = 40)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Flight_Distance', fontsize=10)
plt.subplot(2, 2, 2)
plt.title("Plot 2")
p2 = plt.scatter(independent_var["Age"], independent_var["Arrival_Delay"],c=k_means_output,cmap="viridis", s = 40)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.subplot(2, 2, 3)
plt.title("Plot 3")
p3 = plt.scatter(independent_var["Flight_Distance"], independent_var["Arrival_Delay"],c=k_means_output,cmap="viridis", s = 40)
plt.subplots_adjust(hspace=0.6, wspace=0.6)
plt.xlabel('Flight_Distance', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.figlegend(*p1.legend_elements(), title='Clusters',  loc="lower right")
plt.show()

klabels = np.zeros_like(k_means_output)

#Accuracy of Kmeans
accuracy("Kmeans",3,klabels,k_means_output)

#Outlier Detection Kmeans
def distance_from_center(flight_distance, age, arrival_delay, label):

    #Calculate the Euclidean distance between a data point and the center of its cluster.
    center_flight_distance =  k_means_optimum.cluster_centers_[label,0]
    center_age =  k_means_optimum.cluster_centers_[label,1]
    center_arrival_delay = k_means_optimum.cluster_centers_[label, 2]

    #Euclidean Distance
    distance = np.sqrt((flight_distance - center_flight_distance) ** 2 + (age - center_age) ** 2
                       + (arrival_delay - center_arrival_delay) ** 2)
    return np.round(distance,3)

independent_var['label'] = klabels
independent_var['distance'] = distance_from_center(independent_var["Flight_Distance"],independent_var["Age"],
                              independent_var["Arrival_Delay"],independent_var['label'])

#return 10 most distant points from centroids
outliers_idx = list(independent_var.sort_values('distance', ascending=False).head(10).index)
outliers = independent_var[independent_var.index.isin(outliers_idx)]
print(outliers)

#visualize outliers
plt.suptitle('Kmeans Clusters with Outliers')
plt.subplot(2, 2, 1)
plt.title("Plot 1")
p1 = plt.scatter(independent_var["Age"], independent_var["Flight_Distance"],c=k_means_output,cmap="viridis", s = 40)
plt.scatter(outliers["Age"], outliers["Flight_Distance"], c='aqua', s=100)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Flight_Distance', fontsize=10)
plt.subplot(2, 2, 2)
plt.title("Plot 2")
plt.scatter(independent_var["Age"], independent_var["Arrival_Delay"],c=k_means_output,cmap="viridis", s = 40)
plt.scatter(outliers["Age"], outliers["Arrival_Delay"], c='aqua', s=100)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.subplot(2, 2, 3)
plt.title("Plot 3")
plt.scatter(independent_var["Flight_Distance"], independent_var["Arrival_Delay"],c=k_means_output,cmap="viridis", s = 40)
plt.scatter(outliers["Flight_Distance"], outliers["Arrival_Delay"], c='aqua', s=100)
plt.subplots_adjust(hspace=0.6, wspace=0.6)
plt.xlabel('Flight_Distance', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.figlegend(*p1.legend_elements(), title='Clusters',  loc="lower right")
plt.show()

colors=["red","blue","green","orange"]

#Gaussian Mixture Model and EM
z = StandardScaler()
features = z.fit_transform(independent_var)

#Initialize gmm model
gmm = GaussianMixture(n_components=3,verbose=2)
gmm.fit(features)

GmmClusters = gmm.predict(features)
Gmmlabels = np.zeros_like(GmmClusters)

df['GMMclusters']=GmmClusters
"""data1 = df[df.GMMclusters==0]
data2 = df[df.GMMclusters==1]
data3 = df[df.GMMclusters==2]"""

#Accuracy of GMM
accuracy("GMM",3,Gmmlabels,GmmClusters)

colormap = np.array(['red', 'lime', 'blue'])
plt.suptitle('GMM Clusters')
plt.subplot(2, 2, 1)
plt.title("Plot 1")
p1 = plt.scatter(independent_var["Age"], independent_var["Flight_Distance"],c=GmmClusters,cmap="viridis", s = 40)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Flight_Distance', fontsize=10)
plt.subplot(2, 2, 2)
plt.title("Plot 2")
plt.scatter(independent_var["Age"], independent_var["Arrival_Delay"],c=GmmClusters,cmap="viridis", s = 40)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.subplot(2, 2, 3)
plt.title("Plot 3")
plt.scatter(independent_var["Flight_Distance"], independent_var["Arrival_Delay"],c=GmmClusters,cmap="viridis", s = 40)
plt.subplots_adjust(hspace=0.6, wspace=0.6)
plt.xlabel('Flight_Distance', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.figlegend(*p1.legend_elements(), title='Clusters',  loc="lower right")
plt.show()

#Get the score for each sample
score = gmm.score_samples(independent_var)
#Save score as a column
df['score'] = score
#Get the score threshold for anomaly
pct_threshold = np.percentile(score, 4)
#Print the score threshold
print(f'The threshold of the score is {pct_threshold:.2f}')

#evaluation metrics
sc = silhouette_score(independent_var, klabels)
print("Silhouette Coefficient GMM: ",sc)
ari = adjusted_rand_score(dependent_var, klabels)
print("Adjusted Rand Index GMM:",ari)

#Label the anomalies
df['anomaly_gmm_pct'] = df['score'].apply(lambda x: 1 if x < pct_threshold else 0)

#Visualize the predicted anomalies
plt.subplot(2, 2, 1)
plt.suptitle('GMM Predict Anomalies Using Percentage')
p1 = plt.scatter(independent_var["Age"], independent_var["Flight_Distance"], c=df['anomaly_gmm_pct'], cmap='rainbow')
plt.xlabel('Age', fontsize=10)
plt.ylabel('Flight_Distance', fontsize=10)
plt.title("Plot 1")
plt.subplot(2, 2, 2)
plt.title("Plot 2")
plt.scatter(independent_var["Age"], independent_var["Arrival_Delay"],c=df['anomaly_gmm_pct'],cmap="rainbow", s = 40)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.subplot(2, 2, 3)
plt.title("Plot 3")
plt.scatter(independent_var["Flight_Distance"], independent_var["Arrival_Delay"],c=df['anomaly_gmm_pct'],cmap="rainbow", s = 40)
plt.subplots_adjust(hspace=0.6, wspace=0.6)
plt.xlabel('Flight_Distance', fontsize=10)
plt.ylabel('Arrival_Delay', fontsize=10)
plt.show()