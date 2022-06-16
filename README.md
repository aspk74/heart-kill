# heart-kill : analysing heart disease trend spread through various key indicators and predicting chances of getting a heart disease through various models.

The dataset is a kaggle public dataset about the key indicators of a heart disease. Originally the dataset is taken from CDC and includes data collected till 2020. There were various questions asked about their health to the respondents who reside in the U.S. These were the question inquiring about their physical and mental health like, "Do you experience trouble climbing stairs", etc. The dataset had 279 columns initially which was reduced to 20 columns.
The following steps were followed to reach the required outcome:
  1. The dataset was loaded. 
  2. Necessary variable values(textual) were converted to binary. 
  3. Numeric values were standardised. 
  4. Correlation heatmap was plotted.
  5. Two datasets (X and Y) were declared. X is the dataset conatining all the independent variables. Y is the dataset with the dependent variable.
  6. Split both X and Y into train and test datasets. (80% train, 20% test)
  7. Resolve the problem of uneven data distribution through oversampling.
  8. Train the data with Random Forest Classifier.
  9. Train the data with Naive Bayes Classifier.
  10. The accuracies for RF and NB are 88% and 74% respectively.
  11. Upon removing the oversampling part, accuracies for RF and NB are 90% and 89% respectively.

