import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set graph style
sns.set_style("whitegrid")

# Load dataset
df = pd.read_csv("Unemployment in India.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')

# Remove rows with missing values
df = df.dropna()

# Verify changes
print("\nDataset Information After Cleaning:")
print(df.info())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Region-wise average unemployment rate
region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

# Sort values in descending order
region_avg = region_avg.sort_values(ascending=False)

# Display top 10 states
print("\nTop 10 States by Average Unemployment Rate:")
print(region_avg.head(10))

# Plot graph
plt.figure(figsize=(12,6))
region_avg.head(10).plot(kind='bar')

plt.title('Top 10 States with Highest Average Unemployment Rate')
plt.xlabel('State')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# Create Month-Year column
df['Month'] = df['Date'].dt.to_period('M')

# Calculate average unemployment rate for each month
monthly_trend = df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

# Convert Period to string for plotting
monthly_trend.index = monthly_trend.index.astype(str)

# Plot graph
plt.figure(figsize=(14,6))
plt.plot(monthly_trend.index,
         monthly_trend.values,
         marker='o')

plt.title('Monthly Average Unemployment Rate in India')
plt.xlabel('Month')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# States to analyze
states = ['Tamil Nadu', 'Maharashtra', 'Karnataka']

plt.figure(figsize=(14,6))

for state in states:
    state_data = df[df['Region'] == state]

    plt.plot(
        state_data['Date'],
        state_data['Estimated Unemployment Rate (%)'],
        marker='o',
        label=state
    )

plt.title('Unemployment Rate Over Time for Major States')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()

plt.tight_layout()
plt.show()
# Time-series analysis for major states
states = ['Tamil Nadu', 'Maharashtra', 'Karnataka']

plt.figure(figsize=(14,6))

for state in states:
    state_data = df[df['Region'] == state]

    plt.plot(
        state_data['Date'],
        state_data['Estimated Unemployment Rate (%)'],
        marker='o',
        label=state
    )

plt.title('Unemployment Rate Over Time for Major States')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# Correlation heatmap
correlation_columns = [
    'Estimated Unemployment Rate (%)',
    'Estimated Employed',
    'Estimated Labour Participation Rate (%)'
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(8,6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
# Define COVID split date
covid_date = pd.Timestamp('2020-03-01')

# Split data
pre_covid = df[df['Date'] < covid_date]
post_covid = df[df['Date'] >= covid_date]

# Calculate averages
pre_avg = pre_covid['Estimated Unemployment Rate (%)'].mean()
post_avg = post_covid['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Rate Before COVID:", round(pre_avg, 2))
print("Average Unemployment Rate After COVID:", round(post_avg, 2))

# Plot comparison
plt.figure(figsize=(6,5))

plt.bar(
    ['Pre-COVID', 'Post-COVID'],
    [pre_avg, post_avg]
)

plt.title('Pre-COVID vs Post-COVID Unemployment Rate')
plt.ylabel('Average Unemployment Rate (%)')

plt.tight_layout()
plt.show()
