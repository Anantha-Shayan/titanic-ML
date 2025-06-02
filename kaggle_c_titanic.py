"""Importing modules"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


"""Loading data and creating dataframe"""

train_data = pd.read_csv('train.csv')
train_df = pd.DataFrame(train_data)
test_data = pd.read_csv('test.csv')
test_df = pd.DataFrame(test_data)


"""Feature engineering"""

def feature_engineering(df):

    def substrings_in_string(big_string, substrings):
      """made some changes in this function-- (return np.nan was inside for loop which
        led to adding NaN to all values if if-condition was not satisfied in first
        iteration)"""
      for substring in substrings:
            if substring in big_string:
                return substring
      return np.nan


    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                        'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                        'Don', 'Jonkheer']

    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))


    # replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title=x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title
    # Apply the function to each row (axis=1)
    df['Title'] = df.apply(replace_titles, axis=1)



    # Turning cabin number into Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Cabin'] = df['Cabin'].map(lambda x: str(x))
    df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

    # Creating new family_size column
    df['Family_Size']=df['SibSp']+df['Parch']

    # age*class - an interaction term
    df['Age*Class']=df['Age']*df['Pclass']

    # fare per person
    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
    df.tail()

    # drop 'Ticket' and 'Embarked' as it's of no use
    df = df.drop(columns = 'Ticket', axis = 1)
    df = df.drop(columns = 'Embarked', axis = 1)


    # now drop the columns using which new columns have been created
    remove_from_df = ['Name','Age','SibSp','Parch','Fare','Cabin','Sex','PassengerId']
    for _ in remove_from_df:
      df = df.drop(columns = _, axis = 1)
    #df.head()
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
#print(train_df)
#print(test_df)

#assigning X and y
X_train = train_df.drop('Survived',axis = 1)#.values
X_test = test_df#.values
y_train = train_df['Survived']


"""Feature Scaling"""

#Separate numerical and categorical features
categorical_features = ['Pclass', 'Title', 'Deck']
numerical_features = ['Family_Size', 'Age*Class', 'Fare_Per_Person']

#Feature Scaling for numerical features
scaler = StandardScaler()
X_train_numerical_scaled= scaler.fit_transform(X_train[numerical_features])
X_test_numerical_scaled= scaler.transform(X_test[numerical_features])

# One-hot encode all categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X_train)
X_train_categorical_encoded = encoder.transform(X_train)
X_test_categorical_encoded = encoder.transform(X_test)

# Combine the scaled numerical features and encoded categorical features
X_train_processed = np.concatenate((X_train_numerical_scaled, X_train_categorical_encoded), axis=1)
X_test_processed = np.concatenate((X_test_numerical_scaled, X_test_categorical_encoded), axis=1)

# Using SimpleImputer to handle null values
imputer = SimpleImputer(strategy='most_frequent')
X_train_processed = imputer.fit_transform(X_train_processed)
X_test_processed = imputer.fit_transform(X_test_processed)


"""Model Implementation"""

model = SVC(kernel = 'linear', gamma = 'auto')
model.fit(X_train_processed,y_train)

Y_pred = model.predict(X_test_processed)

true_data = pd.read_csv('gender_submission.csv')
true_df = pd.DataFrame(true_data)
true_df['Predicted Y'] = Y_pred
true_df.to_csv('Submission_result.csv')


"""Evaluation"""

#confusion matrix
cf_matrix = confusion_matrix(true_df['Survived'],true_df['Predicted Y'])
tn, fp, fn, tp = cf_matrix.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*precision*recall/(precision+recall)
print("Accuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1 Score: ",f1_score)
print(tn,fp,fn,tp)

#graphical representation
sns.heatmap(cf_matrix, annot=True)
plt.show()