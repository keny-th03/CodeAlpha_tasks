import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import load_and_clean_data

# Set page config
st.set_page_config(page_title="Unemployment Analysis", layout="wide")

# Load data
unemp, unemp_covid = load_and_clean_data()

# Title
st.title("📊 Unemployment Analysis in India")

# Sidebar filters
st.sidebar.header("Filters")
region_list = sorted(unemp['Region'].unique().tolist())
region_options = ['All'] + region_list

selected = st.sidebar.multiselect(
    "Select Region(s)",
    options=region_options,
    default=['All']
)
# FIX: clean handling of "All"
if not selected or 'All' in selected:
    selected_regions = region_list
else:
    selected_regions = selected

# Filter datasets ONCE
filtered_unemp = unemp[unemp['Region'].isin(selected_regions)]
filtered_unemp_covid = unemp_covid[unemp_covid['Region'].isin(selected_regions)]

# Unemployment Trend Over Time
st.subheader("Unemployment Rate Trend Over Time (Region-wise)")

# Create figure
fig, ax = plt.subplots(figsize=(12,6))

# Color palette
# colors = sns.color_palette("tab10", len(selected_regions))
colors = sns.color_palette("tab10", len(filtered_unemp['Region'].unique()))

# PLot each selected region
# for i, region in enumerate(selected_regions):
for i, region in enumerate(filtered_unemp['Region'].unique()):
    # region_data = unemp[unemp['Region'] == region]
    region_data = filtered_unemp[filtered_unemp['Region'] == region]
    ax.plot(
        region_data['Date'],
        region_data['Estimated Unemployment Rate (%)'],
        label=region,
        color=colors[i],
        linewidth=2
    )

ax.set_xlabel("Year")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_title("Unemployment Rate Trend Over Time")

ax.legend(loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0)
ax.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig)

# Covid-19 Impact
st.subheader("Impact of Covid-19 on Unemployment Rate")

fig2, ax2 = plt.subplots(figsize=(12,6))
colors = sns.color_palette("tab10", len(filtered_unemp_covid['Region'].unique()))

for i, region in enumerate(filtered_unemp_covid['Region'].unique()):
    region_data = filtered_unemp_covid[filtered_unemp_covid['Region'] == region]
    ax2.plot(
        region_data['Date'],
        region_data['Estimated Unemployment Rate (%)'],
        label=region,
        color=colors[i],
        linewidth=2
    )

# Covid start reference
ax2.axvline(pd.to_datetime('2020-03-01'), linestyle='--', color='black', linewidth=2, label='Covid Start')

ax2.set_xlabel("Date")
ax2.set_ylabel("Unemployment Rate (%)")
ax2.set_title("Covid-19 Impact on Unemployment Rate")

ax2.legend(loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0)
ax2.grid(True, linestyle='--', alpha=0.5)
fig2.tight_layout(rect=[0,0,0.85,1])
st.pyplot(fig2)


# Before vs After Covid Comparison
st.subheader("Average Unemployment Rate: Before vs After Covid-19")

# Create a 'Period' column: Before or After Covid
unemp_covid['Period'] = unemp_covid['Date'].apply(lambda x: 'Before Covid' if x < pd.to_datetime('2020-03-01') else 'After Covid')
avg_unemp = unemp_covid.groupby('Period')['Estimated Unemployment Rate (%)'].mean()
fig3, ax3 = plt.subplots(figsize=(4,4))
colors = ['skyblue', 'red']
ax3.pie(
    avg_unemp,
    labels=avg_unemp.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=(0.01,0.01)
)
ax3.set_title("Average Unemployment Rate: Before vs After Covid-19")
st.pyplot(fig3)


# Region-wise Heatmap
st.subheader("Region-wise Average Unemployment Rate (Heatmap)")
heatmap_data = unemp.pivot_table(values='Estimated Unemployment Rate (%)', index='Region', aggfunc='mean')

fig4, ax4 = plt.subplots(figsize=(12,6))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt=".2f", ax=ax4)
ax4.set_title("Region-wise Average Unemployment Rate")
st.pyplot(fig4)


# Seasonal Trend
st.subheader("Seasonal Trend in Unemployment")
fig5, ax5 = plt.subplots(figsize=(10,5))
sns.lineplot(x='Month', y='Estimated Unemployment Rate (%)', data=unemp, ax=ax5)
ax5.set_xlabel("Month")
ax5.set_ylabel("Unemployment Rate (%)")
ax5.set_title("Seasonal Trend in Unemployment")
st.pyplot(fig5)
