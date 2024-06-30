Skeleton Data Classification
----------------------------------------------------------------------------------
(Jupyter Notebook is to be run sequentially)
Note: Please unzip the train and test folder before running the Jupyter notebook


Kaggle Competition (hosted by the MAI faculty THWS University)
https://www.kaggle.com/competitions/learning-of-structured-data-fhws-ws2324

Dataset: Time series movement sequence data set in the .csv format. The sequences are time recordings of movements (joint angles) with different lengths. Labels are given in the file names. 

Goal: Classify the sequences according to the actions: classes - 
"boxing" = 0
"drums" = 1
"guitar" = 2
"rowing" = 3
"violin" = 4

Tasks Performed:

#### Data Loading and Pre-processing:
- Loading and Inspecting Data: Loading skeletal data from a CSV file. The data comprises coordinates representing skeletal joints. 
- Creating XY Coordinate List: A list to organize the x,y coordinates of each joint
- Dropping Unnecessary Columns: Columns that are not required for visualization are dropped from the dataset.
- Defining Connections of the Joints: A dictionary  to represent the connections between different skeletal joints. 


#### Handling Temporal Data
- Representing Temporal Data Using Histograms to get a meaningful representation of the skeletal data.


#### Classification of Histogram Representations: Random Forest Classifier
 - Histogram representations of the skeletal data are classified using a Random Forest Classifier. This algorithm is chosen for its robustness and ability to handle high-dimensional data.


#### Evaluation
- Cross-Validation: The classifier's performance is evaluated using cross-validation, ensuring that the model's performance is consistent across different subsets of the data.
- Test Accuracy: 92%.


