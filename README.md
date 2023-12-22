# Cricket_WorldCup_Prediction-ICC-2023-

# Data Collection:

| Dataset               | Fields                                       |
|-----------------------|----------------------------------------------|
| World Cup 2023        | Team, Group, Previous Appearances, Previous Titles, Previous Finals, Previous Semifinals, Current Rank |
| ICC Rankings          | Position, Team, Points                        |
| Results               | Date, Team_1, Team_2, Winner, Margin, Ground  |
| Fixtures              | Round Number, Date, Location, Team_1, Team_2, Group |
| ODI Match Data        | Match_ID, Season, Start Date, Venue, Innings, Ball, Batting Team, Bowling Team, Striker, Non-Striker, Bowler, Runs Off Bat, Extras, Wides, Noballs, Byes, Legbyes, Penalty, Wicket Type, Player Dismissed, Other Wicket Type, Other Player Dismissed, CricSheet_ID |


The project is divided into four key modules:

# 1.	Match Outcome Prediction Model:


- ### Data Exploration and Preprocessing:
  In the initial stages of the project, we delved into the intricacies of data preprocessing and exploratory data analysis (EDA). Our meticulous approach involved thorough checks for missing values across datasets, ensuring data integrity. We then shifted our focus to gaining a holistic understanding of team performances, visualizing the number of matches played and examining the distribution of winners. A keen eye was also directed towards the historical achievements of teams, portraying the number of titles they've secured through insightful bar plots. As we honed in on the World Cup scenario, we scrutinized and visualized the number of matches won by each team, unraveling patterns and trends. A strategic move was made to refine the dataset for match outcome prediction, involving the judicious removal of non-influential columns.

- ### Training Model:
  To ensure our model learns well, we divided our data into a training set (70%) and a testing set (30%) using scikit-learn's train_test_split. We then delved into three different models:
- Random Forest Model: Constructed a Random Forest Classifier (rf_model) with a fixed random seed (random_state=42). We trained this model using the training data and applied it to make predictions on the testing data.
- Support Vector Machine (SVM) Model: Established an SVM Classifier (svm_model) with a specified random seed and let it learn from our training set.
- Logistic Regression Model: Formulated a Logistic Regression Classifier (lr_model) with a particular random seed and utilized it to make predictions on our testing set.

- ### Model Evaluation:
  In the evaluation of our machine learning models, the RandomForestClassifier, Support Vector Machine (SVC), and Logistic Regression exhibited comparable performance metrics. All three models achieved an accuracy of 0.67, indicating a similar level of overall correctness in their predictions. The F1-scores, measuring the balance between precision and recall, revealed that RandomForestClassifier and SVC share a macro and weighted average F1-Score of 0.40 and 0.63, respectively. LogisticRegression showed a slightly higher macro and weighted average F1-Score of 0.43 and 0.63. Noteworthy are the observed disparities in precision and recall for specific classes, suggesting potential challenges in accurately predicting outcomes for those categories.

# 2.	Team Selection Prediction Model:

- ### Data Exploration and Preprocessing:
  In the context of team selection prediction, the initial data exploration and preprocessing stages focused on ensuring the quality and reliability of the dataset. This involved a meticulous examination of missing values across various datasets, culminating in their identification and subsequent handling. Additionally, the introduction of a 'wickets' column and its systematic updating addressed specific gameplay conditions, contributing to a more nuanced understanding of player performance. The removal of duplicate values further streamlined the dataset, eliminating redundancy and enhancing its overall integrity.

- ### Feature Engineering:
  Feature engineering played a pivotal role in enhancing the dataset for team selection prediction. The creation of cumulative metrics, including wickets, runs, balls faced, batsman strike rate, and bowling strike rate, provided a comprehensive view of individual player performance over multiple matches. By aggregating these key statistics, the dataset evolved to capture the players' sustained contributions and efficiency, offering a more nuanced representation of their capabilities.

- ### Training Model:
  In the training phase of the Linear Regression model for both batsmen and bowlers, the process involved feature extraction and target variable identification. Key features such as average batting strike rate for batsmen and average bowling strike rate for bowlers were selected to capture relevant performance indicators. The subsequent step encompassed the training of the Linear Regression model using the fit function. Through this training process, the model learned the intricate relationships between the chosen features and the target variable, paving the way for accurate predictions of individual player performances in cricket. The emphasis on strike rates in the model underscores its potential efficacy in forecasting the dynamic nature of batting and bowling in the sport.

- ### Model Evaluation:
  The evaluation of the Batsmen and Bowlers Models yields the following metrics:
  #### Batsmen Model Metrics:
- Mean Squared Error (MSE): 0.989
- Root Mean Squared Error (RMSE): 0.994
- Mean Absolute Error (MAE): 0.806
  #### Bowlers Model Metrics:
- Mean Squared Error (MSE): 1.466
- Root Mean Squared Error (RMSE): 1.211
- Mean Absolute Error (MAE): 1.038

  The Batsmen Model exhibits low errors across all metrics, indicating a robust fit to the data and accurate predictions. In comparison, the Bowlers Model, while performing reasonably well, shows slightly higher errors. Overall, both models demonstrate potential in providing valuable insights into individual player performances in cricket, contributing significantly to the objectives of the project. The nuanced variations in error metrics offer a comprehensive view of the models' predictive capabilities for different aspects of the game, emphasizing their utility in advancing cricket analytics.

# 3.	Score Prediction Model:

- ### Data Exploration and Preprocessing:
  The initial steps in data exploration and preprocessing focused on ensuring the dataset's quality and relevance for the score prediction model. Irrelevant columns were identified and dropped to streamline the dataset. The DataFrame was then filtered to include matches involving consistent teams, ensuring a more coherent analysis. To enhance the model's predictive accuracy, rows with a 'ball' column value less than 5.0 were removed, as early-stage data might not be representative of actual match dynamics. Categorical columns, specifically 'batting_team' and 'bowling_team', were encoded using LabelEncoder, facilitating the incorporation of team information into the model.

- ### Feature Engineering:
  To enhance the predictive power of the model, several key features were engineered through a thoughtful process. The creation of the 'Innings_wickets' column, calculating cumulative wickets per inning, provides valuable insights into the evolving dynamics of the match, capturing the ebb and flow of wickets throughout the game. The 'runs_last_5_overs' and 'wickets_last_5_overs' columns were introduced to encapsulate the critical performance indicators in the final phase of the match, shedding light on teams' abilities to capitalize or recover during this crucial period. A correlation heatmap was generated for numerical columns, offering a visual representation of the relationships between various features. This aids in identifying potential multicollinearity and understanding the impact of each variable on the target. One-hot encoding was applied to categorical features ('batting_team' and 'bowling_team') to convert them into a format suitable for machine learning models.

- ### Training Model:
  The dataset was efficiently divided into training and testing sets using an 80-20 split, with 80% allocated to the training set and 20% to the testing set. This balanced distribution ensures a robust model evaluation against unseen data. A Decision Tree Regressor (tree) was employed to capture the nonlinear relationships within the dataset. The model was trained using the features and labels from the training set, allowing it to learn the intricate patterns present in the data. Additionally, a Random Forest Regressor (forest) was implemented to leverage the collective strength of multiple decision trees. This ensemble model was trained on the same training features and labels, harnessing the diversity of individual trees to enhance predictive accuracy.

- ### Model Evaluation:
  The Decision Tree Regressor exhibited a remarkable training score of 99.74%, showcasing its ability to fit the training data almost perfectly. However, this model faced challenges in generalization, as the test score dropped to 72.50%, indicating potential overfitting. For the Decision Tree Regressor, the Mean Absolute Error (MAE) stood at 12.47, the Mean Squared Error (MSE) at 1040.16, and the Root Mean Squared Error (RMSE) at 32.25. These metrics highlight a decent but suboptimal predictive accuracy, especially in comparison to the training set performance. On the other hand, the Random Forest Regression model demonstrated robust generalization with a commendable test score of 86.38%. Despite some errors, this ensemble model outperformed the Decision Tree Regressor, as indicated by lower MAE (13.26), MSE (520.31), and RMSE (22.81) values. These metrics signify improved predictive capabilities and a more accurate representation of real-world cricket match scenarios. In conclusion, the Random Forest Regression model emerges as the preferred choice for cricket match score prediction, striking a balance between fitting the training data well and effectively generalizing to new and unseen data.

# 4.	Player Performance Prediction Model:

- ### Data Exploration and Preprocessing:
  In the data exploration and preprocessing phase for player performance prediction, several crucial steps were undertaken. Entries with an insufficient number of overs (less than 5.0) were removed to ensure a focus on more meaningful and substantial data for analysis. Redundant columns such as 'ID,' 'Country,' 'Bat1,' 'Ground,' 'Start Date,' and 'Match_ID' were dropped, aiming to eliminate irrelevant information and streamline the dataset for more effective analysis. To enhance data consistency, columns like 'Team Runs,' 'Inns,' and 'RPO' were converted to the appropriate data types. Additionally, a thorough check for missing values was conducted, and appropriate actions were taken, which may include either dropping or filling missing values based on the specific context.

- ### Feature Engineering:
  Firstly, dummy variables were generated for categorical columns such as 'Player' and 'Opposition.' This process involved converting categorical variables into binary representations, creating a set of binary columns to represent the various categories within each categorical variable. Subsequently, these dummy variables were concatenated with the original Data Frame, leading to the creation of an expanded and enriched dataset.

- ### Training Model:
  In the training model phase for player performance prediction, a diverse set of regression models was employed, each undergoing a standard train-test split for effective evaluation. Utilizing the train_test_split function from scikit-learn, the dataset was divided into training and testing sets, with an 80-20 split ensuring a robust evaluation process. The following regression models were trained and evaluated:
- Decision Tree Regressor
- K Neighbors Regressor
- XGBRegressor
- RandomForestRegressor
- Linear Regression
- SVR (Support Vector Regression)
This comprehensive approach involves leveraging the strengths of various regression algorithms, allowing for a thorough examination of their predictive capabilities for player performance. The models are now primed for evaluation and comparison based on their individual performances.

- ### Model Evaluation:

<img width="394" alt="image" src="https://github.com/sumedha3/Cricket_WorldCup_Prediction-ICC-2023-/assets/112127474/2da73710-e6f0-492e-9896-c98cb254735f">

  After comparison between the regressors , we can see that Decision Tree Regressor is having the highest accuracy, and which is best for our model. 

