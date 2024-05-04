#!/usr/bin/env python3
# coding: utf-8

# In[65]:


import pandas as pd
import os
import json
import numpy as np

def load_data(folder_path, surface_type):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                for record in json_data['data']:
                    # Extract pose data and flatten it into the record
                    pose_data = record.pop('pose')
                    for key in pose_data:
                        record[key] = pose_data[key]
                    
                    # Add surface type
                    record['surface_type'] = surface_type
                    
                    data.append(record)
    return data

# Paths to the folders
outputs_path = 'outputs'  # Path to the folder with rough surface data
outputs_2_path = 'outputs_2'  # Path to the folder with smooth surface data

# Load the data
rough_data = load_data(outputs_path, 1)  # 1 for rough
smooth_data = load_data(outputs_2_path, 0)  # 0 for smooth

# Combine the data
combined_data = rough_data + smooth_data

# Create a pandas DataFrame
df = pd.DataFrame(combined_data)

# Display the DataFrame
df.head()


# In[66]:


df.describe()


# In[67]:


df.info()


# In[68]:


# Function to calculate the circular mean of angle data
def circular_mean(angles):
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)

# Calculate mean, std, min, max, and count for each list in the 'dists' column
df['dists_mean'] = df['dists'].apply(np.mean)
df['dists_std'] = df['dists'].apply(np.std)
df['dists_min'] = df['dists'].apply(min)
df['dists_max'] = df['dists'].apply(max)
df['dists_count'] = df['dists'].apply(len)

df['angles_mean'] = df['angles'].apply(circular_mean)
df['angles_std'] = df['angles'].apply(lambda x: np.std([np.sin(angle) for angle in x]))
df['angles_min'] = df['angles'].apply(min)
df['angles_max'] = df['angles'].apply(max)
df['angles_count'] = df['angles'].apply(len)

df['surface_type'] = df['surface_type'].astype('category')

# Display the DataFrame
df


# In[69]:


# Remove the 'dists' and 'angles' columns
df = df.drop(columns=['dists', 'angles'])


# In[70]:


pd.set_option('display.max_columns', None)
df.info()


# In[71]:


# Show the direction column info    
df['direction'].value_counts()


# In[72]:


# Set the 'direction' column as a categorical type
df['direction'] = df['direction'].astype('category')
df.info()


# In[73]:


# Info on column "heading"
df['heading']


# In[74]:


# Remove heading column
df = df.drop(columns=['heading'])

df.info()


# In[75]:


df


# In[76]:


df.describe()


# In[77]:


# Chek if there are any null values
df.isnull().sum()


# In[78]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df, hue='surface_type', height=4)
plt.subplots_adjust(top=0.95)
plt.suptitle('Pair Plot', fontsize=20)
plt.show()


# In[79]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Countplot of Surface Type
plt.figure(figsize=(6, 4))
sns.countplot(x='surface_type', data=df)
plt.title('Count of Surface Types')
plt.show()


# In[80]:


# Countplot of Brake by Surface Type
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='surface_type', hue='brake', data=df,
    palette='pastel')
plt.title('Brake Usage by Surface Type')

# Calculate the proportion of counts for each brake category within each surface type
brake_proportions = df.groupby('surface_type')['brake'].value_counts(normalize=True).rename('proportion').reset_index()

# Annotate the plot with the proportion values
for i in range(len(brake_proportions)):
    # Get the proportion value
    proportion = brake_proportions.loc[i, 'proportion']
    # Calculate the number of observations in each category to position the text
    total_count = len(df[df['surface_type'] == brake_proportions.loc[i, 'surface_type']])
    # Calculate the position for the text
    x = brake_proportions.loc[i, 'surface_type']
    y = proportion * total_count
    # Annotate the plot with the proportion value inside a white box
    ax.text(x+0.1, y, f'{proportion:.2f}%', color='black', ha="center", va="center",
            bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.4'))


plt.show()


# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your pandas DataFrame

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Scatterplot of X and Y Coordinates Colored by Surface Type
plt.figure(figsize=(10, 8))
# Plot all points with surface type color coding
sns.scatterplot(x='x', y='y', hue='surface_type', data=df, palette='coolwarm', alpha=0.6)

# Overlay points where braking occurs with a different style
# Filter the DataFrame for rows where brake is applied
braking_points = df[df['brake'] == 1]
# Plot the braking points
# eggplant color is #614051
sns.scatterplot(x='x', y='y', data=braking_points, color='#614051', marker='X', label='Braking', s=100)

# Enhance the plot
plt.title('Spatial Distribution of Surface Types with Braking Points')
plt.legend(loc='upper right')  # Adjust legend position if needed

plt.show()


# In[82]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your pandas DataFrame

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a factorplot to show the relationship between 'direction', 'surface_type', and 'brake'
g = sns.catplot(
    data=df,
    x='direction',
    hue='brake',
    col='surface_type',
    kind='count',
    height=4,
    aspect=1,
    palette='pastel'
)

# Set the axis labels and title
g.set_axis_labels("Direction", "Count")
g.set_titles("Surface Type {col_name}")
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Direction by Surface Type with Brake Counts')

# Annotate each bar with the count
for p in g.axes.flat:
    for bar in p.patches:
        p.annotate(int(bar.get_height()), (bar.get_x()+0.1 + bar.get_width() / 2, int(bar.get_height())),
                   ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                   bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.4'))


# Show the plot
plt.show()
# Show the plot
plt.show()


# In[83]:


df.info()


# In[84]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
label_encoder = LabelEncoder()

# Fit label encoder and return encoded labels
df['direction_encoded'] = label_encoder.fit_transform(df['direction'])
df['horn'] = label_encoder.fit_transform(df['horn'])

# Now 'direction_encoded' is a numerical representation of 'direction' where 'ccw' is 0 and 'cw' is 1
# Remove the original 'direction' column
df = df.drop(columns=['direction'])
df = df.drop(columns=['horn'])


# In[85]:


# Calculate the correlation matrix including the new 'direction_encoded' column
corr_matrix = df.corr()

# Create the correlation matrix plot
plt.figure(figsize=(12,10))  # Increase the figure size if there are many variables
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix with Direction Encoded')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.tight_layout()  # Adjust the layout to fit the plot and labels
# Make text smaller
plt.rc('font', size=1)
plt.show()


# In[86]:


# Plot the correlation just to surface type
plt.figure(figsize=(12,10))  # Increase the figure size if there are many variables
sns.heatmap(corr_matrix[['surface_type']], annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix with Surface Type')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.tight_layout()  # Adjust the layout to fit the plot and labels
plt.show()


# In[87]:


# Drop 'stamp','seq', 'x' and 'y' columns
df = df.drop(columns=['stamp', 'seq'])
# Remove the angles count and change name of dists count in obstacles_count
df = df.drop(columns=['angles_count'])
df = df.rename(columns={'dists_count': 'obstacles_count'})


# In[88]:


# Create the machine learning model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the features and target variable
X = df.drop(columns=['surface_type'])
y = df['surface_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree model with simple hyperparameters
model = DecisionTreeClassifier(max_depth=4, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Display the classification report
print(classification_report(y_test, y_pred))



# In[89]:


# Calculate feature importances
importances = model.feature_importances_

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame(data={'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)
feature_importances


# In[90]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Plot confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
# Plor it
plt.figure(figsize = (3,3))
sns.heatmap(cm, annot = True, cmap = 'plasma')
# label font
plt.xlabel('Predicted label',
           fontdict={'size': 20, 'weight': 'bold'})
plt.ylabel('True label'
           ,fontdict={'size': 20, 'weight': 'bold'})
plt.title('Confusion matrix'
          ,fontdict={'size': 20, 'weight': 'bold'})
# general font
plt.tick_params(axis='both', which='major', labelsize=20)
# font size
plt.rc('font', size=15)
plt.show()


# In[91]:


# Import graphviz
import graphviz
from sklearn.tree import export_graphviz

# Export the decision tree as a dot file
dot_data = export_graphviz(model, out_file=None,
                           feature_names=X.columns,
                           class_names=['0', '1'],
                           filled=True, rounded=True,
                           special_characters=True)

# Render the modified dot file using graphviz
graph = graphviz.Source(dot_data)
# Make the size smaller
graph.render('tree', format='png')
graph


# In[92]:


# Try with a random forest
from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Calculate the accuracy of the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Model Accuracy: {accuracy_rf:.2f}')

# Display the classification report
print(classification_report(y_test, y_pred_rf))


# In[93]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Plot confusion matrix
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
# Plor it
plt.figure(figsize = (3,3))
sns.heatmap(cm, annot = True, cmap = 'plasma')
# label font
plt.xlabel('Predicted label',
           fontdict={'size': 20, 'weight': 'bold'})
plt.ylabel('True label'
           ,fontdict={'size': 20, 'weight': 'bold'})
plt.title('Confusion matrix'
          ,fontdict={'size': 20, 'weight': 'bold'})
# general font
plt.tick_params(axis='both', which='major', labelsize=20)
# font size
plt.rc('font', size=15)
plt.show()


# In[94]:


# Train a Neural Network model
from sklearn.neural_network import MLPClassifier

# Create the Neural Network model
nn_model = MLPClassifier(hidden_layer_sizes=(1000, 50), max_iter=1000, random_state=42)

# Train the model
nn_model.fit(X_train, y_train)

# Make predictions
y_pred_nn = nn_model.predict(X_test)

# Calculate the accuracy of the model
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f'Neural Network Model Accuracy: {accuracy_nn:.2f}')


# In[95]:


# Display the classification report for training
print(classification_report(y_train, nn_model.predict(X_train)))

# Display the classification report for testing
print(classification_report(y_test, nn_model.predict(X_test)))


# In[96]:


# Plot confusion matrix
y_pred = nn_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
# Plor it
plt.figure(figsize = (3,3))
sns.heatmap(cm, annot = True, cmap = 'plasma')
# label font
plt.xlabel('Predicted label',
           fontdict={'size': 20, 'weight': 'bold'})
plt.ylabel('True label'
              ,fontdict={'size': 20, 'weight': 'bold'})
plt.title('Confusion matrix'
            ,fontdict={'size': 20, 'weight': 'bold'})
# general font
plt.tick_params(axis='both', which='major', labelsize=20)
# font size
plt.rc('font', size=15)
plt.show()



