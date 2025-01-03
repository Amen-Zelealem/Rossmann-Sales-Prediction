import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

class Visualyzer:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.train_data = train_data
        self.test_data = test_data

    def check_promotion_distribution(self):
        # Compute proportions of promotions in training and testing datasets
        train_promo_dist = self.train_data['Promo'].value_counts(normalize=True)
        test_promo_dist = self.test_data['Promo'].value_counts(normalize=True)
        train_promo2_dist = self.train_data['Promo2'].value_counts(normalize=True)
        test_promo2_dist = self.test_data['Promo2'].value_counts(normalize=True)

        # Combine datasets for analysis
        self.train_data['Dataset'] = 'Train'
        self.test_data['Dataset'] = 'Test'
        combined_df = pd.concat([self.train_data, self.test_data])

        # Create and print contingency tables for promotions
        promo_contingency = pd.crosstab(combined_df['Promo'], combined_df['Dataset'])
        print("Promo Contingency Table:\n", promo_contingency)

        # Chi-square test for 'Promo'
        chi2_promo, p_promo, _, _ = chi2_contingency(promo_contingency)
        print(f"\nChi-square test for Promo: p-value = {p_promo}")

        # Create and print contingency tables for 'Promo2'
        promo2_contingency = pd.crosstab(combined_df['Promo2'], combined_df['Dataset'])
        print("Promo2 Contingency Table:\n", promo2_contingency)

        # Chi-square test for 'Promo2'
        chi2_promo2, p_promo2, _, _ = chi2_contingency(promo2_contingency)
        print(f"Chi-square test for Promo2: p-value = {p_promo2}")

        # Print promotion distributions
        print("Training Promo Distribution:\n", train_promo_dist)
        print("Testing Promo Distribution:\n", test_promo_dist)
        print("Training Promo2 Distribution:\n", train_promo2_dist)
        print("Testing Promo2 Distribution:\n", test_promo2_dist)

        # Plot distributions with attractive colors
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        sns.barplot(x=train_promo_dist.index, y=train_promo_dist.values, color='teal', alpha=0.7, label='Training')
        sns.barplot(x=test_promo_dist.index, y=test_promo_dist.values, color='salmon', alpha=0.7, label='Testing')
        plt.title('Promo Distribution')
        plt.xlabel('Promo')
        plt.ylabel('Proportion')
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.barplot(x=train_promo2_dist.index, y=train_promo2_dist.values, color='teal', alpha=0.7, label='Training')
        sns.barplot(x=test_promo2_dist.index, y=test_promo2_dist.values, color='salmon', alpha=0.7, label='Testing')
        plt.title('Promo2 Distribution')
        plt.xlabel('Promo2')
        plt.ylabel('Proportion')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def compare_sales_behavior(self):
    # Filter data for open stores and calculate sales around holidays
        df = self.train_data.reset_index()
        data = df[df['Open'] == 1][['Store', 'Date', 'Sales', 'StateHoliday']]

        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by=['Store', 'Date'])
        data['HolidayPeriod'] = np.where(data['StateHoliday'].isin(['a', 'b', 'c']), 'During Holiday', 'Non-Holiday')

        # Identify periods around holidays
        data['BeforeHoliday'] = data['StateHoliday'].shift(-1).isin(['a', 'b', 'c'])
        data['AfterHoliday'] = data['StateHoliday'].shift(1).isin(['a', 'b', 'c'])
        data['HolidayPeriod'] = np.where(data['BeforeHoliday'], 'Before Holiday', data['HolidayPeriod'])
        data['HolidayPeriod'] = np.where(data['AfterHoliday'], 'After Holiday', data['HolidayPeriod'])

        # Group by HolidayPeriod and calculate average sales
        holiday_sales = data.groupby('HolidayPeriod')['Sales'].mean().reset_index()

        # Plot sales behavior with updated colors
        plt.figure(figsize=(10, 4))
        bars = plt.bar(holiday_sales['HolidayPeriod'], holiday_sales['Sales'], color=['#4CAF50', '#FF9800', '#2196F3', '#9C27B0'])

        # Add annotations on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

        # Customize plot
        plt.title('Sales Behavior Before, During, and After Holidays')
        plt.xlabel('Holiday Period')
        plt.ylabel('Average Sales')  # Keep y-axis label
        plt.yticks([])  # Remove tick numbers from y-axis
        plt.show()

    def seasonal_sales_behavior(self, ascending=True):
        # Filter dataset for open stores and calculate average sales by holiday type
        df_open = self.train_data[self.train_data['Open'] == 1].copy()
        df_open['StateHoliday'] = df_open['StateHoliday'].astype(str)
        seasonal_sales = df_open.groupby('StateHoliday')['Sales'].mean().reset_index()

        # Rename holidays for clarity
        seasonal_sales['StateHoliday'] = seasonal_sales['StateHoliday'].replace({
            'a': 'Public Holiday',
            'b': 'Easter Holiday',
            'c': 'Christmas',
            '0': 'No Holiday'
        })

        # Sort based on sales
        seasonal_sales = seasonal_sales.sort_values(by='Sales', ascending=ascending)

        # Define a new color palette
        colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']  # Example custom colors

        # Plot seasonal behavior
        plt.figure(figsize=(8, 4))
        bars = plt.bar(seasonal_sales['StateHoliday'], seasonal_sales['Sales'], color=colors)

        # Annotate bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

        # Customize plot
        plt.title('Seasonal Sales Behavior (Christmas, Easter, etc.)')
        plt.xlabel('Holiday Type')
        plt.ylabel('Average Sales')

        # Remove tick numbers but keep labels
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        plt.yticks([])  # Remove y-axis tick numbers

        plt.show()


    def plot_promo_impact(self):
        # Analyze the impact of promotions on sales and customer counts
        df = self.train_data.reset_index()
        df['Promo'] = df['Promo'].astype(bool)
        df['Promo2'] = df['Promo2'].astype(bool)

        # Calculate averages
        promo_sales_avg = df[df['Promo']]['Sales'].mean()
        non_promo_sales_avg = df[~df['Promo']]['Sales'].mean()
        promo2_sales_avg = df[df['Promo2']]['Sales'].mean()
        non_promo2_sales_avg = df[~df['Promo2']]['Sales'].mean()
        promo_customers_avg = df[df['Promo']]['Customers'].mean()
        non_promo_customers_avg = df[~df['Promo']]['Customers'].mean()
        promo2_customers_avg = df[df['Promo2']]['Customers'].mean()
        non_promo2_customers_avg = df[~df['Promo2']]['Customers'].mean()

        # Create bar charts
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))

        # Sales for Promo
        axs[0, 0].bar(['Promo', 'Non-Promo'], [promo_sales_avg, non_promo_sales_avg], color=['#4C72B8', '#55A868'])
        axs[0, 0].set_title('Average Sales: Promo vs Non-Promo')
        axs[0, 0].set_ylabel('Average Sales')
        axs[0, 0].set_xticks([])  # Remove x-axis tick numbers
        axs[0, 0].set_yticks([])  # Remove y-axis tick numbers

        # Sales for Promo2
        axs[0, 1].bar(['Promo2', 'Non-Promo2'], [promo2_sales_avg, non_promo2_sales_avg], color=['#EBAE3A', '#D55E3B'])
        axs[0, 1].set_title('Average Sales: Promo2 vs Non-Promo2')
        axs[0, 1].set_ylabel('Average Sales')
        axs[0, 1].set_xticks([])  # Remove x-axis tick numbers
        axs[0, 1].set_yticks([])  # Remove y-axis tick numbers

        # Customer Counts for Promo
        axs[1, 0].bar(['Promo', 'Non-Promo'], [promo_customers_avg, non_promo_customers_avg], color=['#4C72B8', '#55A868'])
        axs[1, 0].set_title('Average Customer Count: Promo vs Non-Promo')
        axs[1, 0].set_ylabel('Average Customer Count')
        axs[1, 0].set_xticks([])  # Remove x-axis tick numbers
        axs[1, 0].set_yticks([])  # Remove y-axis tick numbers

        # Customer Counts for Promo2
        axs[1, 1].bar(['Promo2', 'Non-Promo2'], [promo2_customers_avg, non_promo2_customers_avg], color=['#EBAE3A', '#D55E3B'])
        axs[1, 1].set_title('Average Customer Count: Promo2 vs Non-Promo2')
        axs[1, 1].set_ylabel('Average Customer Count')
        axs[1, 1].set_xticks([])  # Remove x-axis tick numbers
        axs[1, 1].set_yticks([])  # Remove y-axis tick numbers

        plt.tight_layout()
        plt.show()

    def _high_impact_stores(self, top_n=10):
        # Analyze stores with the highest impact from promotions
        df = self.train_data.reset_index()
        required_columns = {'Store', 'Promo', 'Promo2', 'Sales', 'Customers'}
        if not required_columns.issubset(df.columns):
            raise KeyError(f"One or more required columns are missing: {required_columns}")

        # Calculate average sales and customers for stores with promotions
        promo_impact = df[df['Promo'] == 1][['Store', 'Sales', 'Customers']].groupby('Store').mean().reset_index()
        promo2_impact = df[df['Promo2'] == 1][['Store', 'Sales', 'Customers']].groupby('Store').mean().reset_index()

        # Merge data for comparison
        common_stores_comparison = pd.merge(promo_impact, promo2_impact, on='Store', suffixes=('_Promo', '_Promo2'))

        # Sort stores by average sales
        promo_impact_sorted = promo_impact.sort_values(by='Sales', ascending=False).head(top_n)
        promo2_impact_sorted = promo2_impact.sort_values(by='Sales', ascending=False).head(top_n)

        # Define color palettes
        promo_colors = sns.color_palette("coolwarm", top_n)
        promo2_colors = sns.color_palette("viridis", top_n)

        # Plotting high-impact stores
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart for Promo
        axs[0].bar(promo_impact_sorted['Store'].astype(str), promo_impact_sorted['Sales'], color=promo_colors)
        axs[0].set_title(f'Top {top_n} High Impact Stores: Promo')
        axs[0].set_xlabel('Store')
        axs[0].set_ylabel('Average Sales')
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].set_xticks([])  # Remove x-axis tick numbers
        axs[0].set_yticks([])  # Remove y-axis tick numbers

        # Bar chart for Promo2
        axs[1].bar(promo2_impact_sorted['Store'].astype(str), promo2_impact_sorted['Sales'], color=promo2_colors)
        axs[1].set_title(f'Top {top_n} High Impact Stores: Promo2')
        axs[1].set_xlabel('Store')
        axs[1].set_ylabel('Average Sales')
        axs[1].tick_params(axis='x', rotation=45)
        axs[1].set_xticks([])  # Remove x-axis tick numbers
        axs[1].set_yticks([])  # Remove y-axis tick numbers

        plt.tight_layout()
        plt.show()

    def analyze_trend(self):
        # Analyze customer behavior trends based on store status
        df = self.train_data.reset_index()
        open_data = df[df['Open'] == 1]
        closed_data = df[df['Open'] == 0]

        # Aggregate data by DayOfWeek
        open_daily_agg = open_data.groupby('DayOfWeek').agg({'Sales': 'mean', 'Customers': 'mean'}).reset_index()
        closed_daily_agg = closed_data.groupby('DayOfWeek').agg({'Sales': 'mean', 'Customers': 'mean'}).reset_index()

        # Plot trends
        fig, ax1 = plt.subplots(figsize=(12, 4))

        # Plot average sales for open and closed states
        color_sales_open = 'blue'
        ax1.set_xlabel('Day of Week')
        ax1.set_ylabel('Average Sales', color=color_sales_open)
        ax1.plot(open_daily_agg['DayOfWeek'], open_daily_agg['Sales'], color=color_sales_open, marker='o', label='Open - Average Sales')
        ax1.plot(closed_daily_agg['DayOfWeek'], closed_daily_agg['Sales'], color='orange', marker='o', linestyle='--', label='Closed - Average Sales')
        
        # Remove x and y axis numbers but keep labels
        ax1.set_xticks([])
        ax1.tick_params(axis='y', labelcolor=color_sales_open)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: ''))  # Corrected lambda function

        # Create a second y-axis for average customers
        ax2 = ax1.twinx()
        color_customers_open = 'green'
        ax2.set_ylabel('Average Customers', color=color_customers_open)
        ax2.plot(open_daily_agg['DayOfWeek'], open_daily_agg['Customers'], color=color_customers_open, marker='o', label='Open - Average Customers')
        ax2.plot(closed_daily_agg['DayOfWeek'], closed_daily_agg['Customers'], color='red', marker='o', linestyle='--', label='Closed - Average Customers')
        
        # Remove y-axis numbers for customers
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: ''))  # Corrected lambda function
        ax2.tick_params(axis='y', labelcolor=color_customers_open)

        # Title and legend
        plt.title('Customer Behavior Trends: Open vs Closed by Day of the Week')
        fig.tight_layout()
        plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.show()
        
    def plot_assortment_sales(self):
        # Analyze sales by assortment type on weekdays and weekends
        data = self.train_data.copy()
        assortment_mapping = {
            'a': 'Basic',
            'b': 'Extra',
            'c': 'Extended'
        }
        data['AssortmentType'] = data['Assortment'].map(assortment_mapping)
        
        # Filter data for weekdays and weekends
        weekday_data = data[data['DayOfWeek'] <= 5]
        weekend_data = data[data['DayOfWeek'] >= 6]
        
        # Calculate average sales for each assortment type
        weekday_sales = weekday_data.groupby('AssortmentType')['Sales'].mean().reset_index()
        weekend_sales = weekend_data.groupby('AssortmentType')['Sales'].mean().reset_index()
        
        # Define new color palette
        colors = ['#4CAF50', '#FF9800', '#2196F3']  # Green, Orange, Blue
        
        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        # Weekday Sales Plot
        sns.barplot(x='AssortmentType', y='Sales', data=weekday_sales, ax=ax[0], hue='AssortmentType', palette=colors, dodge=False)
        ax[0].set_title('Average Weekday Sales by Assortment Type', fontsize=16)
        ax[0].set_xlabel('Assortment Type', fontsize=14)
        ax[0].set_ylabel('Average Sales', fontsize=14)
        ax[0].set_xticks([])  # Remove x-axis numbers
        ax[0].set_yticks([])  # Remove y-axis numbers

        # Weekend Sales Plot
        sns.barplot(x='AssortmentType', y='Sales', data=weekend_sales, ax=ax[1], hue='AssortmentType', palette=colors, dodge=False)
        ax[1].set_title('Average Weekend Sales by Assortment Type', fontsize=16)
        ax[1].set_xlabel('Assortment Type', fontsize=14)
        ax[1].set_ylabel('Average Sales', fontsize=14)
        ax[1].set_xticks([])  # Remove x-axis numbers
        ax[1].set_yticks([])  # Remove y-axis numbers
        
        plt.tight_layout()
        plt.show()