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
