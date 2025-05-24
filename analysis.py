#Cleaning and Handling Missing Values
# python
# Copy
# Edit
# Check for missing values
missing_values = df.isnull().sum()

# Fill numeric columns with median, categorical with mode
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
#2. Feature Selection and Engineering
# python
# Copy
# Edit
# Create a new feature: Loss per user
df['Loss per User ($)'] = df['Financial Loss (in Million $)'] * 1e6 / df['Number of Affected Users']

# Select important features (example)
selected_features = df[['Country', 'Year', 'Attack Type', 'Target Industry', 
                        'Financial Loss (in Million $)', 'Number of Affected Users', 
                        'Loss per User ($)']]
#3. Ensuring Data Integrity and Consistency
# python
# Copy
# Edit
# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize categorical values
df['Attack Type'] = df['Attack Type'].str.strip().str.title()
df['Country'] = df['Country'].str.strip().str.title()
#4. Summary Statistics and Insights
# python
# Copy
# Edit
summary_stats = df.describe(include='all')
top_attacks = df['Attack Type'].value_counts()
mean_loss_by_country = df.groupby('Country')['Financial Loss (in Million $)'].mean()
#5. Identifying Patterns, Trends, and Anomalies
# python
# Copy
# Edit
# Trends over years
trend_by_year = df.groupby('Year')[['Financial Loss (in Million $)', 'Number of Affected Users']].sum()

# Detect anomalies: Extremely high loss per user
anomalies = df[df['Loss per User ($)'] > df['Loss per User ($)'].quantile(0.95)]
#6. Handling Outliers and Data Transformations
# python
# Copy
# Edit
# Handle outliers using IQR
numeric_cols = ['Financial Loss (in Million $)', 'Number of Affected Users']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Apply log transformation to skewed columns
import numpy as np
df['Log Loss'] = np.log1p(df['Financial Loss (in Million $)'])
#7. Initial Visual Representation of Key Findings
# python
# Copy
# Edit
import matplotlib.pyplot as plt
import seaborn as sns

# Bar plot: Top 5 attack types
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Attack Type', order=df['Attack Type'].value_counts().head(5).index)
plt.title('Top 5 Attack Types')
plt.xticks(rotation=45)
plt.show()

# Heatmap: Correlation between numerical features
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Line plot: Financial Loss over Years
plt.figure(figsize=(10, 6))
trend_by_year['Financial Loss (in Million $)'].plot(marker='o')
plt.title('Yearly Financial Loss')
plt.ylabel('Financial Loss (in Million $)')
plt.grid(True)
plt.show()
