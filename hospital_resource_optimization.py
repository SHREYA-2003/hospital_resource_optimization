# load dataset
# df_train = pd.read_csv('./Data/train_data.csv')
# df_test = pd.read_csv('./Data/test_data.csv')
df_train = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')
df_test = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')

# To check complete dataset (column wise and row wise)
pd.set_option('display.max_columns', None)    # this is to display all the columns in the dataframe
# pd.set_option('display.max_rows', None)       # this is to display all the rows in the dataframe

#Analyze data
# To check data from training dataset
df_train.head()

# to check details of training dataset
df_train.info()

#The 04 numerical columns are Ids Hospital_code, City_Code_Hospital, City_Code_Patient and Bed Grade, So we have to convert them in Strings in both Training and Test datasets
#There are some missing values in 02 columns Bed Grade and City_Code_Patient
df_train['Hospital_code'] = df_train['Hospital_code'].astype('str')
df_train['City_Code_Hospital'] = df_train['City_Code_Hospital'].astype('str')
df_train['Bed Grade'] = df_train['Bed Grade'].astype('Int64')
df_train['City_Code_Patient'] = df_train['City_Code_Patient'].astype('str')

df_train.info()
df_train[['City_Code_Patient']].value_counts()
df_test['Hospital_code'] = df_test['Hospital_code'].astype('str')
df_test['City_Code_Hospital'] = df_test['City_Code_Hospital'].astype('str')
df_test['Bed Grade'] = df_test['Bed Grade'].astype('Int64')
df_test['City_Code_Patient'] = df_test['City_Code_Patient'].astype('str')
df_test.info()
# To check data from test dataset
df_test.head()
# to check details of test dataset
df_test.info()
# check the shape of training and test dataset
print('shape of training data set: ', df_train.shape)
print('shape of test data set: ', df_test.shape)
df_train[['patientid']].nunique()

#It is noticed that after converting City_Code_Patient in string, the missing values are replaced with 'nan' string
#There are more than one records or rows of one patient patientid

# Find the Structure or size of Dataset and check the Descriptive Analysis
print('Shape of dataset : ',df_train.shape)
df_train.describe()
# Find the Structure or size of Dataset and check the Descriptive Analysis
print('Shape of dataset : ',df_test.shape)
df_test.describe()
# Checking for missing values in Training dataset
df_train.isnull().sum().any()
df_train.loc[:, df_train.isna().any()].isna().sum().sort_values(ascending=False)
df_test.loc[:, df_test.isna().any()].isna().sum().sort_values(ascending=False)
# Check null/missing values inside dataset in descending order
df_train.isnull().sum().sort_values(ascending=False)
# Plot missing values in Training dataset
plt.figure(figsize = (16,6))
sns.heatmap(df_train.isnull(), yticklabels= False, cbar= False, cmap='viridis')
# Plot missing values in Test dataset
plt.figure(figsize = (16,6))
sns.heatmap(df_test.isnull(), yticklabels= False, cbar= False, cmap='viridis')
# Checking for missing values in test dataset
df_test.isnull().sum().any()
# find missing values in Training dataset according to their percentage
missing_perc = (df_train.isnull().sum()/len(df_train)*100).sort_values(ascending=False)
missing_perc[missing_perc != 0]
# Plot the null values in Training Dataset by their percentage
missing_perc[missing_perc != 0].plot(kind='bar')
plt.xlabel("Columns")
plt.ylabel("Percentage")
plt.title('Percentage of Missing Values in each columns')
# find missing values in Test dataset according to their percentage
missing_perc_test = (df_test.isnull().sum()/len(df_test)*100).sort_values(ascending=False)
missing_perc_test[missing_perc_test != 0]
# Plot the null values in Test Dataset by their percentage
missing_perc_test[missing_perc_test != 0].plot(kind='bar')
plt.xlabel("Columns")
plt.ylabel("Percentage")
plt.title('Percentage of Missing Values in each columns')
# check any duplication
df_train.duplicated(subset=df_train.columns.difference(['case_id'])).any()
df_train.duplicated(subset=df_train.columns.difference(['case_id'])).sum()
df_train.duplicated(subset=df_train.columns).sum()
df_test.duplicated(subset=df_test.columns.difference(['case_id'])).sum()
df_train[df_train.duplicated(subset=df_train.columns.difference(['case_id']), keep=False)]

#Remove Anamolies from Dataset
df_train.drop_duplicates(subset=df_train.columns.difference(['case_id']), inplace=True)
print(df_train.shape)
df_train.head()
# first we have to check the dataset
df_train['Stay'].value_counts().sort_index()
df_train[['Stay','Age']].value_counts().sort_index()
df_train['Age'].value_counts().sort_index()
df_train.columns
df_train['Severity of Illness'].value_counts()
df_train['Department'].value_counts()
df_train['Ward_Type'].value_counts()
df_train['Bed Grade'].value_counts().sort_index()
#Check Unique values in all the columns along with maximum and minimum values in the Numerical columns
# Check Unique values, data type of each column and Minimum and Maximum values of Numerical columns
for column in df_train.columns:
    unique_values = df_train[column].unique()
    type_value = df_train[column].dtype
    if len(unique_values) > 10:
        unique_values = unique_values[:10]
    total_unique_values = df_train[column].nunique()
    print(f"Data Type of {column}: {type_value}")
    print(f"Total Unique values in {column}: {total_unique_values}")
    if (df_train[column].dtype == 'int64') or (df_train[column].dtype == 'float64'):
       print(f"Minimum value: {df_train[column].min()},   Maximum value: {df_train[column].max()}")
    print(f"Unique values in {column}: {unique_values}\n")
# Lets separate Numerical and categorical columns to visualize properly

num_col_train = df_train.select_dtypes(include=np.number).columns.difference(['case_id','patientid'])
cat_col_train = df_train.select_dtypes(include=['object','category']).columns
# Visualize Numerical columns

for col in num_col_train:
    print(col)
    print(f"Minimum value: {df_train[col].min()},   Maximum value: {df_train[col].max()}")  
    print('Skew :', round(df_train[col].skew(),2))
    plt.figure(figsize = (15,4))
    plt.subplot(1,2,1)
    df_train[col].hist(grid = False)
    # sns.histplot(data=df_train, x=df_train[col], kde=True)
    plt.ylabel('Count')
    plt.title(col)
    plt.subplot(1,2,2)
    plt.title(col)
    sns.boxplot(x=df_train[col])
    plt.show()
# Print Categorical columns and their value counts
print(cat_col_train)
for i,col in enumerate(cat_col_train):
    print(df_train[col].value_counts())
    print('\n')
# Distribution of all Categorical columns
cat_col_train = df_train.select_dtypes(include=['object','category']).columns
print(f"Categorical Columns are : {cat_col_train}")
# Visualize Categorical columns 
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(18, 26))
# Increase vertical spacing
plt.subplots_adjust(hspace=0.5)
# Set the supertitle
fig.suptitle('Bar plot for all categorical variables in the dataset with their Status\n', fontsize=20)
# Adjust the spacing between the supertitle and subplots
plt.subplots_adjust(top=0.95)
# Iterate over the columns and create count plots
for i, column in enumerate(cat_col_train):
    row = i // 2
    col = i % 2
    sns.countplot(ax=axes[row, col], x=column, data=df_train) # , hue='Stay')
   
    total_count = len(df_train[column])

    for p in axes[row,col].patches:
        percentage = f'{100 * p.get_height() / total_count:.1f}%'
        x_pos = p.get_x() + p.get_width() / 2
        y_pos = p.get_height()
        axes[row,col].annotate(percentage, (x_pos, y_pos), ha='center', va='bottom')
# Adjust the layout and display the plots
plt.tight_layout()
plt.show()
# Visualization of correlation in numerical columns
plt.figure(figsize=(12,8))
sns.heatmap(df_train[num_col_train].corr(),cbar = True, cmap='coolwarm', annot=True)
import plotly.express as px
# piechart

df_pie = df_train['Stay'].value_counts().reset_index()
df_pie.columns = ['Stay', 'count']
fig_pie = px.pie(df_pie, values='count', names='Stay', title="Pie Plot showing distribution of the Length of Stay in the Hospital") #, category_orders={'Stay':'0-10'})

fig_pie.show()
#Maximum cases of patients (27.5%) staying in Hospital for 21-30 days, at 2nd number more cases of patients (24.5%) staying for 11-20 days, at 3rd number cases of patients (17.3%) staying are 31-40 days as shown in above pie chart.
df_train.columns
# create sunburst plot on the dataset

# Create a sunburst plot

fig = px.sunburst(df_train, 
                  path=['Severity of Illness', 'Type of Admission','Department','Ward_Type', 'Stay'], 
                  values='Bed Grade' , color='Type of Admission', title="Chart shows the distribution of the Status ")
fig.update_layout(width = 800, height= 800)
# Show the plot
fig.show()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier 
#import grid search cv for cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss, make_scorer
# train test split of dataset in python
from sklearn.model_selection import train_test_split

# Splitting the DataFrame into features (X) and target variable (y)
X_train = df_train.drop(['Stay', 'case_id','patientid'], axis=1)  
y_train = df_train['Stay']  

X_test = df_test.drop(['case_id','patientid'], axis=1) 

# Splitting the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Lets separate Numerical and categorical columns to visualize properly
num_col = X_train.select_dtypes(include=np.number).columns
cat_col = X_train.select_dtypes(include=['object']).columns
# Create a dictionaries of list of models to evaluate performance with hyperparameters
models = { 
          'DecisionTreeClassifier' : (DecisionTreeClassifier(), {'criterion' : ['gini'],'max_depth': [None], 'splitter': ['best']}),
          'RandomForestClass' : (RandomForestClassifier(n_jobs= -1), {'criterion' : ['gini'],'n_estimators': [10,100], 'max_depth': [None]}),
          'AdaBoostClassifier' : (AdaBoostClassifier(), {'n_estimators': [10,100], 'algorithm': ['SAMME', 'SAMME.R']}),
         # 'GradientBoostingClassifier' : (GradientBoostingClassifier(), {'criterion' : ['friedman_mse','squared_error'], 'n_estimators': [10]}),
         # 'XGBClassifier' : (XGBClassifier(), {'n_estimators': [10], 'learning_rate': [0.1]}),          
          }

# Define the column transformer for preprocessing
preprocessing = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), num_col),  # Replace with actual numeric column names
        ('categorical', OneHotEncoder(), cat_col)  # Replace with actual categorical column name
    ])

# LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

for name, (model, params) in models.items():
    # create a pipline
    # pipeline = GridSearchCV(model, params, cv=5)
    pipeline = Pipeline([
     ('preprocess', preprocessing),
     ('Imputer', SimpleImputer()),
     ('classify', GridSearchCV(model, params, cv=5, verbose=3))
    ])
    # fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # make prediction from each model
    y_pred = pipeline.predict(X_test)
    
    y_pred_prob = pipeline.predict_proba(X_test)
    # print the performing metric
    # Calculate accuracy
    print(f"Model Name: {name}")
    print("Best Parameters: ", pipeline.named_steps['classify'].best_params_)
    print("Best Score: ", pipeline.named_steps['classify'].best_score_)
    print("Best Estimator: ", pipeline.named_steps['classify'].best_estimator_)
