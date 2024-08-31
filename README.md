# SyriaTel Customer Churn Project
## Business Understanding
### Project overview
The project aims to build a predictive model to identify customers who are likely to churn (stop doing business) from SyriaTel, a telecommunications company. Churn prediction is crucial for the company because retaining existing customers is often more cost-effective than acquiring new ones. By predicting churn, the company can proactively take steps to retain customers, thereby reducing revenue loss and improving customer satisfaction.

### Problem Statement
The core problem is to develop a binary classification model that accurately predicts whether a customer will churn. Given that customer churn can significantly impact the company's profitability, identifying patterns and factors that contribute to churn is essential. The challenge lies in analyzing the available customer data to uncover insights that can help in formulating strategies to minimize churn.

### Project Objectives
Analyze customer data to identify key features and patterns that contribute to churn. Develop and validate a binary classification model that predicts customer churn with high accuracy. Provide actionable insights and recommendations for SyriaTel to implement targeted retention strategies based on the model's findings. Evaluate the business impact of the predictive model by estimating potential revenue saved through early identification and intervention for at-risk customers.

## Data Understanding
The dataset used in this project originates from a telecommunications company, SyriaTel, and is designed to predict customer churn whether a customer will leave the company in the near future. The dataset consists of various features that describe customer behavior, account information, and service usage.

### Data Characteristics:
Numerical Features: The dataset contains various numerical features representing customer usage and service interactions, which are crucial in identifying patterns related to churn.
Categorical Features: Features like state, international_plan, and voice_mail_plan are categorical and may provide insights when analyzed for churn patterns.
Class Imbalance: The churn variable is typically imbalanced, with fewer customers churning compared to those who stay. This imbalance needs to be addressed during modeling to avoid biased predictions.

### Preprocessing Considerations:

Multicollinearity: Some features were removed earlier due to multicollinearity, ensuring that the remaining features contribute independently to the model.
Feature Engineering: New features or transformations might be needed to enhance the model’s predictive power.
Normalization and Scaling: Given the range of values across different features, normalization and scaling are applied to ensure that all features contribute equally to the model.
This data understanding forms the foundation for building an effective model, helping to identify which features are most influential in predicting customer churn and guiding the preprocessing steps to prepare the data for modeling.

## Data Cleaning & Preparation
The initial exploration of the dataset reveals some positive aspects:

No Missing Values: The dataset is complete, with no missing entries across any of the columns.

However, one area of concern is the area_code column. Early observations show only three distinct area codes, despite the dataset covering a diverse range of states. The describe function also supports this, indicating that only three unique area codes are present. This inconsistency raises questions about the reliability of the area_code data. If further investigation confirms this anomaly, it may be necessary to drop the area_code column from the analysis due to its potential inaccuracy.
The analysis confirms that the area_code data is unreliable, as only three distinct area codes are present across 51 states. This lack of diversity in area codes is inconsistent with what we would expect from a dataset covering multiple states. Given this, the area_code column does not provide meaningful information and will be dropped from further analysis. Additionally, the phone_number column will also be removed, as it does not contribute to understanding or predicting customer churn.

The heatmap identifies perfect correlations (value of 1.00) between several pairs of features: total_day_minutes and total_day_charge, total_eve_minutes and total_eve_charge, total_night_minutes and total_night_charge, as well as total_intl_minutes and total_intl_charge. This is expected since charges are typically calculated based on the number of minutes spent on calls. However, this introduces multicollinearity into our model, which can distort the impact of individual variables and lead to overfitting. To mitigate this, we will drop the total_x_minutes columns (total_day_minutes, total_eve_minutes, total_night_minutes, and total_intl_minutes) from our dataset. Retaining only the corresponding total_x_charge columns should provide the same information without redundancy.

![Correlation with Churn Heatmap](SyriaTel-customer-project/heatmap diagram.png)

## Data Analysis
We begin by analyzing the relationship between various features both categorical and numerical and customer churn to identify key factors that influence a customer's decision to leave the company.

Categorical Features: The categorical features analyzed included state, international_plan, and voice_mail_plan. For each feature, we plotted the distribution of churned versus non-churned customers using count plots. These visualizations help us understand how different customer attributes correlate with churn. For instance, we could observe whether certain states or service plans are more associated with higher churn rates.

Numerical Features: For numerical features, such as account_length, number_vmail_messages, total_day_calls, total_eve_calls, total_night_calls, total_intl_calls, and customer_service_calls, we created both box plots and violin plots. These visualizations provided insights into the spread and distribution of these features among churned and non-churned customers. Box plots allowed us to see the quartiles and potential outliers, while violin plots provided a deeper look at the density distribution of the data.

Train-Test Split: The dataset was split into training and testing sets, with 80% of the data allocated for training and 20% for testing. This split enabled us to train our models on one portion of the data and evaluate their performance on another to ensure generalization.

Model Evaluation: We tested multiple models to determine which would best fit our data:

Logistic Regression: Served as our baseline model. We trained it on the training data and assessed its performance.
Decision Tree: We explored this model while addressing overfitting concerns by tuning its hyperparameters.
XGBoost: As a more advanced model, XGBoost was evaluated for its performance and ability to handle the dataset's complexity.
The goal was to identify the model that offered the best performance metrics, such as accuracy, recall, and precision, and was well-suited for predicting customer churn.

![Train Confusion Matrix](SyriaTel-customer-project/confusion matrix 1.png) 

Extremely High Accuracy: The overall accuracy of the ResNet50 model is exceptionally high at approximately 99.98%. This indicates that the model is making correct predictions for a vast majority of instances.
Perfect Precision: The precision is 1.0, which is the highest possible value. This means that whenever the model predicts a positive instance (a customer churning), it is always correct.
Very High Recall: The recall is also very high at 99.91%. This means that the model is capturing a significant number of actual positive instances (customers who churned).

![Test Confusion Matrix](SyriaTel-customer-project/confusion matrix 2.png)

### Feature Importance 
In our data analysis, we explored the relationship between various features both categorical and numerical and customer churn to identify key factors influencing a customer's decision to leave the company. We analyzed categorical features like 'state,' 'international_plan,' and 'voice_mail_plan' using count plots, revealing how different categories correlate with churn. For numerical features such as 'account_length,' 'number_vmail_messages,' and 'customer_service_calls,' we employed box plots and violin plots to compare their distributions between churned and non-churned customers. This allowed us to uncover patterns and potential indicators of churn. Following this, we developed predictive models, starting with a baseline Logistic Regression and moving to more complex models like Decision Trees and XGBoost, evaluating each for performance and overfitting. Feature importance was then assessed using the XGBoost model, where we identified and visualized the top 20 features most predictive of churn, offering valuable insights into customer behavior.

![Top 20 Features of Importance](SyriaTel-customer-project/top 20 features.png)

## Conclusion
Using the data provided, we were able to create a model with 82% recall, meaning of the customers who are going to leave, we are able to identify 82% of them. We were able to do this while maintining a high accuracy of 95%.

Based on the feature importance we can determine that churn can be influenced by:

If the customer has an international plan Voice mail plan and number of voicemail messages Customer Service Calls Total day charge

## Recommendation
It's evident that many states appear in the top 20 most important features. Based on these insights, we recommend the following actions:

Investigate the Needs of International Plan Customers: Consider whether there's a shift toward online communication tools for international users, such as Skype, Discord, Google Chat, or FaceTime. Explore the possibility of offering more robust data plans to accommodate this trend.

Audit Customer Service Calls: Ensure that customers are receiving adequate support. If the model identifies a customer at risk of churning, it might be beneficial to review their recent interactions with customer service to address any unresolved issues.

Review Rate Competitiveness: Given that total day charges are a significant factor in our model, it’s important to verify that pricing is competitive, particularly in states with higher churn rates.

Moving forward, we should also consider the following steps:

Address Area Code Data Issues: Improving the accuracy of area code information will allow us to analyze geographic churn with greater precision. Although area codes don’t perfectly correspond to customer locations, they can still provide useful approximations.

Enhance Call Center Data: Currently, we only know how many times a customer has called. It would be beneficial to gather more detailed information, such as customer satisfaction, call duration, and the need for escalation.

Expand Customer Account Information: Beyond call data, we should also collect information on texting and data usage. Additionally, understanding the number of lines on an account could provide valuable insights.
