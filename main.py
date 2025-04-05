import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from shapely.geometry import Point

# Set page configuration
st.set_page_config(
    page_title="White Land Tax Visualization - Saudi Arabia",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        color: #34495e;
        text-align: center;
        margin-bottom: 30px;
    }
    .dashboard-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .stat-box {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">White Land Tax Visualization Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Saudi Arabia Land Tax Analysis and Insights</div>', unsafe_allow_html=True)

# Function to load and process data
@st.cache_data
def load_data():
    # In a real application, you might load this from a database or file
    # For this example, we'll use the data from the provided CSV
    # Adjust the path as needed
    file_path = r'1743420287-White_Land_Tax_Sample_Data (1).csv'
    df = pd.read_csv(file_path)
    
    # Convert coordinates to numeric values
    df['Land co-ordinates(WGS84)'] = df['Land co-ordinates(WGS84)'].astype(str)
    # # df['Latitude'] = df['Land co-ordinates(WGS84)'].apply(lambda x: float(x.split(',')[0].strip()))
    # df['Latitude'] = df['Land co-ordinates(WGS84)'].apply(lambda x: float(x.split(',')[0].strip()))
    # df['Longitude'] = df['Land co-ordinates(WGS84)'].apply(lambda x: float(x.split(',')[1].strip()))
    df[['Latitude', 'Longitude']] = df['Land co-ordinates(WGS84)'].str.split(',', expand=True).astype(float)
    
    # Create point geometry for geospatial analysis
    df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Convert acquisition date and valuation date to datetime
    df['Acquisition date'] = pd.to_datetime(df['Acquisition date'])
    df['Valuation date'] = pd.to_datetime(df['Valuation date'])
    
    # Calculate tax amount (2.5% of current value for plots that are not exempt)
    df['Tax Rate (%)'] = 0.0
    df.loc[df['Exemption from WLT (Yes or No)'] != 'Yes', 'Tax Rate (%)'] = 2.5
    df.loc[(df['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'] == 'Undeveloped') & 
        (df['Exemption from WLT (Yes or No)'] != 'Yes'), 'Tax Rate (%)'] = 2.5
    df.loc[(df['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'] == 'Partially Developed') & 
        (df['Exemption from WLT (Yes or No)'] != 'Yes'), 'Tax Rate (%)'] = 1.25
    
    df['Tax Amount (SAR)'] = df.apply(
        lambda row: 0 if row['Exemption from WLT (Yes or No)'] == 'Yes' 
        else row['Current value of owned land (SAR) as per business plan'] * row['Tax Rate (%)'] / 100,
        axis=1
    )
    
    # Extract discount information
    discounts = {
        'Telecommunication': {'radius': 500, 'discount': 3.5},
        'Road': {'radius': 500, 'discount': 25},
        'Electricity': {'radius': 500, 'discount': 25},
        'Water Drainage (Rain)': {'radius': 500, 'discount': 3.7},
        'Water': {'radius': 500, 'discount': 11.4},
        'Sewerage System': {'radius': 500, 'discount': 11.8},
        'Religious Services': {'radius': 1000, 'discount': 1},
        'Security Services': {'radius': 5000, 'discount': 5},
        'Health Services': {'radius': 5000, 'discount': 1},
        'Education Services': {'radius': 2000, 'discount': 5},
        'Mountain': {'radius': None, 'discount': 30},
        'Valley': {'radius': None, 'discount': 30},
        'Water Body': {'radius': None, 'discount': 30}
    }
    
    return df, discounts, gdf

# Function to calculate tax discounts based on infrastructure
def calculate_discounts(row, discounts):
    total_discount = 0
    for service, values in discounts.items():
        col_name = service if service not in ['Mountain', 'Valley', 'Water Body'] else f'Has {service}'
        if col_name in row.index and not pd.isna(row[col_name]) and row[col_name] < 5:  # 5 is considered close enough for infrastructure
            total_discount += values['discount']
    
    # Cap total discount at 50%
    return min(total_discount, 50)

# Function to create infrastructure impact visualization
def create_infrastructure_chart(discounts):
    # Create a chart showing how infrastructure affects tax discounts
    discount_data = []
    
    for service, values in discounts.items():
        if service not in ['Mountain', 'Valley', 'Water Body']:
            discount_data.append({
                'Service': service,
                'Maximum Discount (%)': values['discount'],
                'Effective Radius (m)': values['radius']
            })
    
    discount_df = pd.DataFrame(discount_data)
    fig = px.bar(
        discount_df,
        x='Service',
        y='Maximum Discount (%)',
        title='Tax Discounts by Infrastructure Type',
        color='Maximum Discount (%)',
        hover_data=['Effective Radius (m)'],
        color_continuous_scale=px.colors.sequential.Viridis
    )
    return fig

# Function to create tax savings simulator
def tax_savings_simulator(df, discounts):
    st.subheader("Tax Savings Simulator")
    
    # Allow user to select a plot
    plot_options = list(df['Plot number'].astype(str).unique())
    selected_plot = st.selectbox("Select a plot to simulate development impact:", plot_options)
    
    # Get selected plot data
    plot_data = df[df['Plot number'].astype(str) == selected_plot].iloc[0]
    
    # Status selection
    current_status = plot_data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)']
    st.write(f"Current Status: {current_status}")
    
    new_status_options = ['Undeveloped', 'Partially Developed', 'Developed']
    new_status = st.selectbox("Simulated Development Status:", new_status_options, 
                             index=new_status_options.index(current_status) if current_status in new_status_options else 0)
    
    # Infrastructure improvements
    st.write("Select infrastructure improvements:")
    col1, col2 = st.columns(2)
    
    improvements = {}
    for i, (service, values) in enumerate(discounts.items()):
        if service not in ['Mountain', 'Valley', 'Water Body']:
            column = col1 if i % 2 == 0 else col2
            service_col = service if service in plot_data.index else f"{service} Distance"
            current_value = plot_data.get(service_col, "N/A")
            
            with column:
                improvements[service] = st.checkbox(
                    f"Improve {service} (Current: {current_value}, Max Discount: {values['discount']}%)",
                    value=False
                )
    
    # Calculate current tax
    current_tax_rate = plot_data['Tax Rate (%)']
    property_value = plot_data['Current value of owned land (SAR) as per business plan'] if not pd.isna(plot_data['Current value of owned land (SAR) as per business plan']) else 0
    current_tax = property_value * current_tax_rate / 100
    
    # Calculate new tax
    new_tax_rate = 0.0
    if new_status == 'Undeveloped' and plot_data['Exemption from WLT (Yes or No)'] != 'Yes':
        new_tax_rate = 2.5
    elif new_status == 'Partially Developed' and plot_data['Exemption from WLT (Yes or No)'] != 'Yes':
        new_tax_rate = 1.25
    
    # Apply infrastructure discounts
    infrastructure_discount = 0
    for service, improved in improvements.items():
        if improved:
            infrastructure_discount += discounts[service]['discount']
    
    # Cap discount at 50%
    infrastructure_discount = min(infrastructure_discount, 50)
    final_tax_rate = max(0, new_tax_rate * (1 - infrastructure_discount / 100))
    new_tax = property_value * final_tax_rate / 100
    
    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Annual Tax (SAR)", f"{current_tax:,.2f}")
    col2.metric("New Annual Tax (SAR)", f"{new_tax:,.2f}")
    col3.metric("Tax Savings (SAR)", f"{current_tax - new_tax:,.2f}", 
               delta=f"{((current_tax - new_tax) / current_tax * 100) if current_tax > 0 else 0:.1f}%")
    
    # Tax breakdown
    st.subheader("Tax Calculation Breakdown")
    st.write(f"Property Value: {property_value:,.2f} SAR")
    st.write(f"Base Tax Rate for {new_status}: {new_tax_rate:.2f}%")
    st.write(f"Infrastructure Discount: {infrastructure_discount:.2f}%")
    st.write(f"Final Tax Rate: {final_tax_rate:.2f}%")
    st.write(f"Annual Tax Amount: {new_tax:,.2f} SAR")

# Load the data
data, discounts, gdf = load_data()

# Sidebar
st.sidebar.header("Filters")

# City filter
# cities = sorted(data['City'].unique(), key=str)
# # cities = sorted(data['City'].unique())
data['City'] = data['City'].astype(str)
cities = sorted(data['City'].unique())
selected_city = st.sidebar.multiselect("Select Cities", cities, default=cities)

# Land use filter
land_uses = sorted(data['Land Use'].unique())
selected_land_use = st.sidebar.multiselect("Select Land Use", land_uses, default=land_uses)

# Land status filter
data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)']=data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'].astype(str)
land_statuses = sorted(data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'].unique())
selected_land_status = st.sidebar.multiselect("Select Land Status", land_statuses, default=land_statuses)

# Exemption filter
data['Exemption from WLT (Yes or No)']= data['Exemption from WLT (Yes or No)'].astype(str)
exemption_status = sorted(data['Exemption from WLT (Yes or No)'].unique())
selected_exemption = st.sidebar.multiselect("Select Exemption Status", exemption_status, default=exemption_status)

# Filter data based on selections
filtered_data = data[
    (data['City'].isin(selected_city)) &
    (data['Land Use'].isin(selected_land_use)) &
    (data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'].isin(selected_land_status)) &
    (data['Exemption from WLT (Yes or No)'].isin(selected_exemption))
]

# Info message if data is filtered out
if len(filtered_data) == 0:
    st.warning("No data available for the selected filters. Please adjust your selection.")
else:
    # Dashboard layout with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Map Visualization", "Tax Analysis", "Tax Simulator", "Detailed Data"])
    
    # TAB 1: Overview
    with tab1:
        # Key metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_plots = len(filtered_data)
            st.metric("Total Land Plots", f"{total_plots:,}")
        
        with col2:
            exempt_plots = len(filtered_data[filtered_data['Exemption from WLT (Yes or No)'] == 'Yes'])
            exempt_percentage = (exempt_plots / total_plots) * 100 if total_plots > 0 else 0
            st.metric("Exempt Plots", f"{exempt_plots:,} ({exempt_percentage:.1f}%)")
        
        with col3:
            total_area = filtered_data['Land plot size Owned (sqm)'].sum()
            st.metric("Total Land Area (sqm)", f"{total_area:,.0f}")
        
        with col4:
            total_tax = filtered_data['Tax Amount (SAR)'].sum()
            st.metric("Total Tax Revenue (SAR)", f"{total_tax:,.0f}")
        
        # Distribution by Land Use
        st.subheader("Land Distribution by Land Use")
        land_use_counts = filtered_data['Land Use'].value_counts().reset_index()
        land_use_counts.columns = ['Land Use', 'Count']
        
        # Create pie chart
        fig1 = px.pie(
            land_use_counts, 
            values='Count', 
            names='Land Use',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Land Status & Tax Analysis
        st.subheader("Land Status vs Tax Revenue")
        col1, col2 = st.columns(2)
        
        with col1:
            # Land Status Distribution
            status_counts = filtered_data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'].value_counts().reset_index()
            status_counts.columns = ['Land Status', 'Count']
            
            fig2 = px.bar(
                status_counts,
                x='Land Status',
                y='Count',
                color='Land Status',
                text='Count'
            )
            fig2.update_layout(title="Land Status Distribution", height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Tax by Land Status
            tax_by_status = filtered_data.groupby('Land Status ( Developed / Undeveloped / Partially Developed/WIP)')['Tax Amount (SAR)'].sum().reset_index()
            tax_by_status.columns = ['Land Status', 'Tax Amount (SAR)']
            
            fig3 = px.bar(
                tax_by_status,
                x='Land Status',
                y='Tax Amount (SAR)',
                color='Land Status',
                text_auto='.2s'
            )
            fig3.update_layout(title="Tax Revenue by Land Status", height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        # City Analysis
        st.subheader("Analysis by City")
        city_metrics = filtered_data.groupby('City').agg(
            Total_Plots=('Registration number', 'count'),
            Total_Area=('Land plot size Owned (sqm)', 'sum'),
            Total_Tax=('Tax Amount (SAR)', 'sum'),
            Avg_Value=('Current value of owned land (SAR) as per business plan', 'mean')
        ).reset_index()
        
        fig4 = px.bar(
            city_metrics,
            x='City',
            y='Total_Tax',
            color='City',
            text_auto='.2s',
            hover_data=['Total_Plots', 'Total_Area', 'Avg_Value']
        )
        fig4.update_layout(title="Tax Revenue by City", height=500)
        st.plotly_chart(fig4, use_container_width=True)
    
    # TAB 2: Map Visualization
    with tab2:
        st.subheader("Land Plots Map")
        
        # Create a folium map centered at Saudi Arabia
        saudi_center = [24.7136, 46.6753]  # Approximate center of Saudi Arabia
        m = folium.Map(location=saudi_center, zoom_start=6, tiles="OpenStreetMap")
        
        # Add tile layers
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('openstreetmap').add_to(m)
        
        # Add markers for each land plot
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in filtered_data.iterrows():
            # Define color based on development status and exemption status
            if row['Exemption from WLT (Yes or No)'] == 'Yes':
                color = 'green'  # Exempt
            elif row['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'] == 'Developed':
                color = 'blue'   # Developed (low or no tax)
            elif row['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'] == 'Partially Developed':
                color = 'orange' # Partially developed (medium tax)
            else:
                color = 'red'    # Undeveloped (high tax)
            
            # Create popup content
            popup_content = f"""
                <b>Registration #:</b> {row['Registration number']}<br>
                <b>Plot #:</b> {row['Plot number']}<br>
                <b>Land Use:</b> {row['Land Use']}<br>
                <b>Status:</b> {row['Land Status ( Developed / Undeveloped / Partially Developed/WIP)']}<br>
                <b>City:</b> {row['City']}<br>
                <b>Area:</b> {row['Land plot size Owned (sqm)']:,.0f} sqm<br>
                <b>Current Value:</b> {row['Current value of owned land (SAR) as per business plan']:,.0f} SAR<br>
                <b>Exempt:</b> {row['Exemption from WLT (Yes or No)']}<br>
                <b>Tax Rate:</b> {row['Tax Rate (%)']:.2f}%<br>
                <b>Tax Amount:</b> {row['Tax Amount (SAR)']:,.0f} SAR
            """
            
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=10,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(marker_cluster)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display the map
        folium_static(m, width=1200, height=600)
        
        # Display map legend
        st.markdown("""
        <div style="background-color:white; padding:10px; border-radius:5px;">
            <h4>Map Legend</h4>
            <p><span style="color:green">●</span> Exempt Land Plots (No WLT)</p>
            <p><span style="color:blue">●</span> Developed Land Plots (Low/No Tax)</p>
            <p><span style="color:orange">●</span> Partially Developed Land Plots (Medium Tax)</p>
            <p><span style="color:red">●</span> Undeveloped Land Plots (Full Tax)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Heat map of land values
        st.subheader("Land Value Distribution")
        
        # Create a heatmap using plotly
        fig_heatmap = px.density_mapbox(
            filtered_data, 
            lat='Latitude', 
            lon='Longitude', 
            z='Current value of owned land (SAR) as per business plan', 
            radius=15,
            center=dict(lat=24.7136, lon=46.6753),
            zoom=5,
            mapbox_style="open-street-map",
            title="Heat Map of Land Values"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # TAB 3: Tax Analysis
    with tab3:
        st.subheader("White Land Tax Analysis")
        
        # Tax Overview
        col1, col2 = st.columns(2)
        
        with col1:
            # Tax Distribution by Land Use
            tax_by_land_use = filtered_data.groupby('Land Use')['Tax Amount (SAR)'].sum().reset_index()
            tax_by_land_use = tax_by_land_use.sort_values('Tax Amount (SAR)', ascending=False)
            
            fig5 = px.bar(
                tax_by_land_use,
                x='Land Use',
                y='Tax Amount (SAR)',
                color='Land Use',
                text_auto='.2s'
            )
            fig5.update_layout(title="Tax Revenue by Land Use", height=400)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Tax Exemption Reasons
            if 'Yes' in selected_exemption:
                exempt_data = filtered_data[filtered_data['Exemption from WLT (Yes or No)'] == 'Yes']
                exemption_reasons = exempt_data['Reason for Exemption'].value_counts().reset_index()
                exemption_reasons.columns = ['Reason', 'Count']
                
                fig6 = px.pie(
                    exemption_reasons,
                    values='Count',
                    names='Reason',
                    hole=0.4
                )
                fig6.update_layout(title="Exemption Reasons", height=400)
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("No exempt properties selected. Adjust filters to include exempt properties.")
        
        # Tax Potential Analysis
        st.subheader("Tax Revenue Potential Analysis")
        
        # Calculate potential tax if all properties were taxable
        total_potential_tax = data['Current value of owned land (SAR) as per business plan'].sum() * 0.025
        actual_tax = data['Tax Amount (SAR)'].sum()
        
        # Tax gap
        tax_gap = total_potential_tax - actual_tax
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Potential Tax Revenue (SAR)", f"{total_potential_tax:,.0f}")
        with col2:
            st.metric("Actual Tax Revenue (SAR)", f"{actual_tax:,.0f}")
        with col3:
            st.metric("Tax Gap (SAR)", f"{tax_gap:,.0f}")
        
        # Create a comparison chart
        tax_comparison = pd.DataFrame({
            'Category': ['Potential Tax', 'Actual Tax', 'Tax Gap'],
            'Amount (SAR)': [total_potential_tax, actual_tax, tax_gap]
        })
        
        fig7 = px.bar(
            tax_comparison,
            x='Category',
            y='Amount (SAR)',
            color='Category',
            text_auto='.2s'
        )
        fig7.update_layout(height=400)
        st.plotly_chart(fig7, use_container_width=True)
        
        # Tax Efficiency by City
        st.subheader("Tax Efficiency by City")
        
        city_tax_analysis = data.groupby('City').agg(
            Potential_Tax=('Current value of owned land (SAR) as per business plan', lambda x: sum(x) * 0.025),
            Actual_Tax=('Tax Amount (SAR)', 'sum')
        ).reset_index()
        
        city_tax_analysis['Tax_Efficiency'] = (city_tax_analysis['Actual_Tax'] / city_tax_analysis['Potential_Tax']) * 100
        
        fig8 = px.bar(
            city_tax_analysis,
            x='City',
            y='Tax_Efficiency',
            color='City',
            text=city_tax_analysis['Tax_Efficiency'].round(1).astype(str) + '%'
        )
        fig8.update_layout(
            title="Tax Collection Efficiency by City (%)",
            yaxis_title="Efficiency (%)",
            height=400
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        # Infrastructure Impact Chart
        st.subheader("Infrastructure Impact on Tax Discounts")
        st.plotly_chart(create_infrastructure_chart(discounts), use_container_width=True)
    
    # TAB 4: Tax Simulator
    with tab4:
        tax_savings_simulator(data, discounts)
    
    # TAB 5: Detailed Data
    with tab5:
        st.subheader("Detailed Land Plot Data")
        
        # Display data table with selected columns
        columns_to_display = [
            'Registration number', 'Plot number', 'City', 'Land Use', 
            'Land Status ( Developed / Undeveloped / Partially Developed/WIP)', 
            'Land plot size Owned (sqm)', 'Current value of owned land (SAR) as per business plan',
            'Exemption from WLT (Yes or No)', 'Tax Rate (%)', 'Tax Amount (SAR)'
        ]
        
        st.dataframe(filtered_data[columns_to_display], use_container_width=True)
        
        # Allow downloading the filtered data
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name=f"white_land_tax_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.markdown("© 2025 White Land Tax Dashboard for Saudi Arabia | All rights reserved")