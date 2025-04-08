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
from shapely.geometry import Polygon, LineString, Point
from geopy.distance import geodesic
import folium.plugins as plugins
import branca.colormap as cm
import json
from matplotlib.colors import LinearSegmentedColormap

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
    file_path = r'Test.csv'
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

# # Land use filter
# land_uses = sorted(data['Land Use'].unique())
# selected_land_use = st.sidebar.multiselect("Select Land Use", land_uses, default=land_uses)

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
    # (data['Land Use'].isin(selected_land_use)) &
    (data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'].isin(selected_land_status)) &
    (data['Exemption from WLT (Yes or No)'].isin(selected_exemption))
]


def create_tax_zones(gdf):
    """
    Create tax zones based on land characteristics and location for specific cities.
    Returns GeoDataFrame with zone polygons for Al Hofuf, Jeddah, Dammam, and Riyadh.
    """
    # Create empty list to store all zone data
    zones = []
    
    # Define city centers (approximate coordinates)
    city_centers = {
        'Riyadh': [24.7136, 46.6753],
        'Jeddah': [21.4858, 39.1925],
        'Dammam': [26.4207, 50.0888],
        'Al Hofuf': [25.3801, 49.5859]
    }
    
    # Create zones for each city
    for city, center in city_centers.items():
        if city in gdf['City'].unique() or any(gdf['City'].str.contains(city, case=False)):
            # Red zone (high tax) - innermost urban area
            red_zone = {
                'zone_type': 'Red',
                'city': city,
                'description': f'High Tax Zone ({city})',
                'tax_rate': 2.5,
                'geometry': Polygon([
                    (center[1] - 0.1, center[0] - 0.1),
                    (center[1] + 0.1, center[0] - 0.1),
                    (center[1] + 0.1, center[0] + 0.1),
                    (center[1] - 0.1, center[0] + 0.1),
                    (center[1] - 0.1, center[0] - 0.1)
                ])
            }
            zones.append(red_zone)
            
            # Amber zone (medium tax) - middle zone
            amber_zone = {
                'zone_type': 'Amber',
                'city': city,
                'description': f'Medium Tax Zone ({city})',
                'tax_rate': 1.25,
                'geometry': Polygon([
                    (center[1] - 0.2, center[0] - 0.2),
                    (center[1] + 0.2, center[0] - 0.2),
                    (center[1] + 0.2, center[0] + 0.2),
                    (center[1] - 0.2, center[0] + 0.2),
                    (center[1] - 0.2, center[0] - 0.2)
                ])
            }
            zones.append(amber_zone)
            
            # Green zone (low/exempt tax) - outermost zone
            green_zone = {
                'zone_type': 'Green',
                'city': city,
                'description': f'Low Tax/Exempt Zone ({city})',
                'tax_rate': 0.0,
                'geometry': Polygon([
                    (center[1] - 0.3, center[0] - 0.3),
                    (center[1] + 0.3, center[0] - 0.3),
                    (center[1] + 0.3, center[0] + 0.3),
                    (center[1] - 0.3, center[0] + 0.3),
                    (center[1] - 0.3, center[0] - 0.3)
                ])
            }
            zones.append(green_zone)
    
    # Create GeoDataFrame
    zones_gdf = gpd.GeoDataFrame(zones, crs="EPSG:4326")
    return zones_gdf


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
            filtered_data['Total land plot size (sqm)'] =filtered_data['Total land plot size (sqm)'].astype(float)
            total_area = filtered_data['Total land plot size (sqm)'].sum()
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
            tax_by_status.columns = ['Land Status ( Developed / Undeveloped / Partially Developed/WIP)', 'Tax Amount (SAR)']
            
            fig3 = px.bar(
                tax_by_status,
                x='Land Status ( Developed / Undeveloped / Partially Developed/WIP)',
                y='Tax Amount (SAR)',
                color='Land Status ( Developed / Undeveloped / Partially Developed/WIP)',
                text_auto='.2s'
            )
            fig3.update_layout(title="Tax Revenue by Land Status", height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        # City Analysis
        st.subheader("Analysis by City")
        city_metrics = filtered_data.groupby('City').agg(
            Total_Plots=('Registration number', 'count'),
            Total_Area=('Total land plot size (sqm)', 'sum'),
            Total_Tax=('Tax Amount (SAR)', 'sum'),
            # Avg_Value=('Current value of owned land (SAR) as per business plan', 'mean')
        ).reset_index()
        
        fig4 = px.bar(
            city_metrics,
            x='City',
            y='Total_Tax',
            color='City',
            text_auto='.2s',
            hover_data=['Total_Plots', 'Total_Area']
        )
        fig4.update_layout(title="Tax Revenue by City", height=500)
        st.plotly_chart(fig4, use_container_width=True)
    
 # Modify the map visualization tab to include city-specific zones and analysis
    with tab2:
            st.subheader("Land Plots Map with Tax Zones")
            
            # Create tax zones
            tax_zones = create_tax_zones(gdf)
            
            # Create a folium map centered at Saudi Arabia
            saudi_center = [24.7136, 46.6753]  # Approximate center of Saudi Arabia
            m = folium.Map(location=saudi_center, zoom_start=6, tiles="OpenStreetMap")
            
            # Add tile layers
            folium.TileLayer('cartodbpositron').add_to(m)
            folium.TileLayer('openstreetmap').add_to(m)
            
            # Add city selector for the map
            cities_in_data = sorted(filtered_data['City'].unique())
            selected_map_city = st.selectbox(
                "Focus map on city:",
                ["All Cities"] + list(cities_in_data)
            )
            
            if selected_map_city != "All Cities":
                # If a specific city is selected, center the map on that city
                for city, center in {
                    'Riyadh': [24.7136, 46.6753],
                    'Jeddah': [21.4858, 39.1925],
                    'Dammam': [26.4207, 50.0888],
                    'Al Hofuf': [25.3801, 49.5859]
                }.items():
                    if city.lower() in selected_map_city.lower():
                        m.location = center
                        m.zoom_start = 10
                        break
            
            # Define zone colors
            zone_colors = {
                'Red': 'red',
                'Amber': 'orange',
                'Green': 'green'
            }
            
            # Create feature groups for each city's zones
            city_zone_groups = {}
            for city in tax_zones['city'].unique():
                city_zone_groups[city] = folium.FeatureGroup(name=f"{city} Tax Zones", show=selected_map_city == "All Cities" or city.lower() in selected_map_city.lower())
            
            # Add zone polygons to map
            for idx, zone in tax_zones.iterrows():
                zone_geojson = gpd.GeoSeries([zone['geometry']]).to_json()
                
                folium.GeoJson(
                    zone_geojson,
                    name=f"{zone['city']} - {zone['zone_type']} Zone",
                    style_function=lambda x, color=zone_colors[zone['zone_type']], opacity=0.4: {
                        'fillColor': color,
                        'color': color,
                        'weight': 2,
                        'fillOpacity': opacity
                    },
                    tooltip=f"{zone['city']} - {zone['zone_type']} Zone - {zone['description']} (Tax Rate: {zone['tax_rate']}%)"
                ).add_to(city_zone_groups[zone['city']])
            
            # Add feature groups to map
            for city, feature_group in city_zone_groups.items():
                feature_group.add_to(m)
            
            # Calculate which zone each plot is in and its distance to zone boundaries
            filtered_data['Zone'] = None
            filtered_data['Zone_City'] = None
            filtered_data['Zone_Tax_Rate'] = None
            
            # Create distance columns for each zone
            for city in tax_zones['city'].unique():
                for zone_type in ['Red', 'Amber', 'Green']:
                    filtered_data[f'Distance_to_{city}_{zone_type}'] = None
            
            # Process each plot
            for idx, row in filtered_data.iterrows():
                point = Point(row['Longitude'], row['Latitude'])
                
                # Find which zone this point is in
                for zone_idx, zone in tax_zones.iterrows():
                    if point.within(zone['geometry']):
                        filtered_data.at[idx, 'Zone'] = zone['zone_type']
                        filtered_data.at[idx, 'Zone_City'] = zone['city']
                        filtered_data.at[idx, 'Zone_Tax_Rate'] = zone['tax_rate']
                        break
                
                # Calculate distances to all zone boundaries by city
                for city in tax_zones['city'].unique():
                    city_zones = tax_zones[tax_zones['city'] == city]
                    
                    for zone_idx, zone in city_zones.iterrows():
                        # Convert to a LineString to get the boundary
                        boundary = LineString(zone['geometry'].exterior.coords)
                        
                        # Calculate distance to boundary (in meters)
                        dist = point.distance(boundary) * 111000  # Approximate conversion from degrees to meters
                        filtered_data.at[idx, f'Distance_to_{city}_{zone["zone_type"]}'] = dist
            
            # Create different marker clusters for each city
            marker_clusters = {}
            for city in filtered_data['City'].unique():
                marker_clusters[city] = plugins.MarkerCluster(name=f"{city} Plots")
                marker_clusters[city].add_to(m)
            
            # Default cluster for plots without a matching city
            default_cluster = plugins.MarkerCluster(name="Other Plots")
            default_cluster.add_to(m)
            
            # Add markers for each land plot
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
                
                # Get zone and distance information
                zone_info = ""
                if not pd.isna(row['Zone']):
                    zone_info = f"<b>Tax Zone:</b> {row['Zone']} ({row['Zone_City']})<br>"
                    zone_info += f"<b>Zone Tax Rate:</b> {row['Zone_Tax_Rate']}%<br>"
                
                # Get distance information for the plot's city
                distance_info = "<b>Distance to Zone Boundaries:</b><br>"
                plot_city = row['City']
                
                # Match the plot city with our defined cities
                matching_city = None
                for city in tax_zones['city'].unique():
                    if city.lower() in plot_city.lower():
                        matching_city = city
                        break
                
                if matching_city:
                    for zone_type in ['Red', 'Amber', 'Green']:
                        distance_col = f'Distance_to_{matching_city}_{zone_type}'
                        if distance_col in row.index and not pd.isna(row[distance_col]):
                            distance_info += f"- {matching_city} {zone_type} Zone: {row[distance_col]:.2f} meters<br>"
                else:
                    # If no matching city is found, show distances to all cities
                    for city in tax_zones['city'].unique():
                        distance_info += f"<b>{city} Zones:</b><br>"
                        for zone_type in ['Red', 'Amber', 'Green']:
                            distance_col = f'Distance_to_{city}_{zone_type}'
                            if distance_col in row.index and not pd.isna(row[distance_col]):
                                distance_info += f"- {zone_type}: {row[distance_col]:.2f} meters<br>"
                
                # Create popup content
                popup_content = f"""
                                        **Registration #:** {row['Registration number']}
                                        **Plot #:** {row['Plot number']}
                                        **Land Use:** {row['Land Use']}
                                        **Status:** {row['Land Status ( Developed / Undeveloped / Partially Developed/WIP)']}
                                        **City:** {row['City']}
                                        **Area:** {str(row['Total land plot size (sqm)'])} sqm
                                        **Current Value:** {(row['Current value of owned land (SAR) as per business plan'])} SAR
                                        **Exempt:** {row['Exemption from WLT (Yes or No)']}
                                        **Tax Rate:** {float(row['Tax Rate (%)']):,.2f}%
                                        **Tax Amount:** {float(row['Tax Amount (SAR)']):,.0f} SAR
                                        {zone_info}
                                        {distance_info}
                                    """
                
                circle_marker = folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=8,
                    popup=folium.Popup(popup_content, max_width=400),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                )
                
                # Add to appropriate cluster
                added_to_cluster = False
                for city_name, cluster in marker_clusters.items():
                    if city_name.lower() in row['City'].lower():
                        circle_marker.add_to(cluster)
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    circle_marker.add_to(default_cluster)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Display the map
            folium_static(m, width=1200, height=600)
            
            # Display map legend
            st.markdown("""
            <div style="background-color:white; padding:10px; border-radius:5px;">
                <h4>Map Legend</h4>
                <h5>Plot Status:</h5>
                <p><span style="color:green">●</span> Exempt Land Plots (No WLT)</p>
                <p><span style="color:blue">●</span> Developed Land Plots (Low/No Tax)</p>
                <p><span style="color:orange">●</span> Partially Developed Land Plots (Medium Tax)</p>
                <p><span style="color:red">●</span> Undeveloped Land Plots (Full Tax)</p>
                
                <h5>Tax Zones:</h5>
                <p><span style="color:red; background-color:rgba(255,0,0,0.2); padding:2px 8px;">■</span> Red Zone - High Tax (2.5%)</p>
                <p><span style="color:orange; background-color:rgba(255,165,0,0.2); padding:2px 8px;">■</span> Amber Zone - Medium Tax (1.25%)</p>
                <p><span style="color:green; background-color:rgba(0,128,0,0.2); padding:2px 8px;">■</span> Green Zone - Low/Exempt Tax (0%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add zone analysis
            st.subheader("Tax Zone Analysis")
            
            # City selector for analysis
            analysis_city = st.selectbox(
                "Select city for zone analysis:",
                ["All Cities"] + list(cities_in_data)
            )
            
            # Filter data for selected city
            if analysis_city != "All Cities":
                city_data = filtered_data[filtered_data['City'] == analysis_city]
            else:
                city_data = filtered_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Count plots in each zone
                if 'Zone' in city_data.columns:
                    zone_counts = city_data['Zone'].value_counts().reset_index()
                    zone_counts.columns = ['Zone', 'Count']
                    
                    # Add a "Not in Zone" category for plots that aren't in any defined zone
                    if zone_counts['Zone'].isna().any():
                        na_count = zone_counts[zone_counts['Zone'].isna()]['Count'].iloc[0]
                        zone_counts = zone_counts[~zone_counts['Zone'].isna()]
                        zone_counts = pd.concat([zone_counts, pd.DataFrame([{'Zone': 'Not in Zone', 'Count': na_count}])])
                    
                    fig_zone = px.pie(
                        zone_counts,
                        values='Count',
                        names='Zone',
                        color='Zone',
                        color_discrete_map={'Red': 'red', 'Amber': 'orange', 'Green': 'green', 'Not in Zone': 'gray'},
                        title=f"Land Plots Distribution by Tax Zone ({analysis_city})"
                    )
                    st.plotly_chart(fig_zone, use_container_width=True)
                else:
                    st.info("No zone data available for the selected city.")
            
            with col2:
                # Tax distribution by zone
                if 'Zone' in city_data.columns and 'Tax Amount (SAR)' in city_data.columns:
                    tax_by_zone = city_data.groupby('Zone')['Tax Amount (SAR)'].sum().reset_index()
                    
                    # Add a "Not in Zone" category if needed
                    if tax_by_zone['Zone'].isna().any():
                        na_tax = tax_by_zone[tax_by_zone['Zone'].isna()]['Tax Amount (SAR)'].iloc[0]
                        tax_by_zone = tax_by_zone[~tax_by_zone['Zone'].isna()]
                        tax_by_zone = pd.concat([tax_by_zone, pd.DataFrame([{'Zone': 'Not in Zone', 'Tax Amount (SAR)': na_tax}])])
                    
                    fig_tax = px.bar(
                        tax_by_zone,
                        x='Zone',
                        y='Tax Amount (SAR)',
                        color='Zone',
                        color_discrete_map={'Red': 'red', 'Amber': 'orange', 'Green': 'green', 'Not in Zone': 'gray'},
                        title=f"Tax Revenue by Zone ({analysis_city})",
                        text_auto='.2s'
                    )
                    st.plotly_chart(fig_tax, use_container_width=True)
                else:
                    st.info("No tax data available for zones in the selected city.")
            
            # Show distance analysis
            st.subheader("Distance Analysis")
            
            # Create a consolidated distance analysis dataframe
            if analysis_city != "All Cities":
                # Find matching city from our defined cities
                matching_city = None
                for city in tax_zones['city'].unique():
                    if city.lower() in analysis_city.lower():
                        matching_city = city
                        break
                
                if matching_city:
                    # Get distance columns for this city
                    distance_cols = [col for col in city_data.columns if f'Distance_to_{matching_city}' in col]
                    
                    if distance_cols:
                        # Create a boxplot of distances
                        distance_df = city_data[distance_cols].copy()
                        
                        # Rename columns for better display
                        distance_df.columns = [col.replace(f'Distance_to_{matching_city}_', '') for col in distance_cols]
                        
                        # Melt for plotting
                        distance_melted = pd.melt(
                            distance_df,
                            value_vars=distance_df.columns,
                            var_name='Zone Boundary',
                            value_name='Distance (m)'
                        )
                        
                        fig_box = px.box(
                            distance_melted,
                            x='Zone Boundary',
                            y='Distance (m)',
                            color='Zone Boundary',
                            color_discrete_map={'Red': 'red', 'Amber': 'orange', 'Green': 'green'},
                            title=f"Distance Distribution to Zone Boundaries in {analysis_city} (meters)"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.info(f"No distance data available for {analysis_city}.")
                else:
                    st.info(f"No defined tax zones match {analysis_city}.")
            else:
                # Show a summary table for all cities
                city_zone_summary = []
                
                for city in tax_zones['city'].unique():
                    city_distance_cols = [col for col in filtered_data.columns if f'Distance_to_{city}' in col]
                    
                    if city_distance_cols:
                        city_summary = {
                            'City': city,
                        }
                        
                        for zone in ['Red', 'Amber', 'Green']:
                            zone_col = f'Distance_to_{city}_{zone}'
                            if zone_col in filtered_data.columns:
                                city_summary[f'Avg Distance to {zone} Zone (m)'] = filtered_data[zone_col].mean()
                        
                        city_zone_summary.append(city_summary)
                
                if city_zone_summary:
                    city_summary_df = pd.DataFrame(city_zone_summary)
                    st.dataframe(city_summary_df.round(2), use_container_width=True)
                else:
                    st.info("No distance data available.")

                    # Add this section to the end of your map visualization tab to provide deeper insights
# on the relationship between zone distances and tax rates

            # Continue from previous code in tab2
            st.subheader("Zone Distance vs. Tax Analysis")

            # Create selector for city
            dist_analysis_city = st.selectbox(
                "Select city for distance-tax analysis:",
                ["All Cities"] + list(cities_in_data),
                key="dist_analysis_selector"
            )

            if dist_analysis_city != "All Cities":
                plot_data = filtered_data[filtered_data['City'] == dist_analysis_city]
            else:
                plot_data = filtered_data

            # Find which city's zones are most relevant for this data
            relevant_cities = []
            for city in tax_zones['city'].unique():
                cols = [col for col in plot_data.columns if f'Distance_to_{city}' in col]
                if cols and not plot_data[cols].isna().all().all():
                    relevant_cities.append(city)

            if relevant_cities:
                selected_zone_city = st.selectbox(
                    "Select zone city for analysis:",
                    relevant_cities
                )
                
                # Get zone distance columns for this city
                zone_distance_cols = [col for col in plot_data.columns if f'Distance_to_{selected_zone_city}_' in col]
                
                if zone_distance_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create scatter plot of distance vs. tax rate
                        for zone_type in ['Red', 'Amber', 'Green']:
                            distance_col = f'Distance_to_{selected_zone_city}_{zone_type}'
                            
                            if distance_col in zone_distance_cols:
                                # Create temporary df without NaN values
                                temp_df = plot_data[[distance_col, 'Tax Rate (%)']].dropna()
                                
                                if not temp_df.empty:
                                    fig_scatter = px.scatter(
                                        temp_df,
                                        x=distance_col,
                                        y='Tax Rate (%)',
                                        title=f"Distance to {zone_type} Zone vs. Tax Rate",
                                        color_discrete_sequence=[zone_colors[zone_type]],
                                        trendline="ols"
                                    )
                                    
                                    fig_scatter.update_layout(
                                        xaxis_title=f"Distance to {selected_zone_city} {zone_type} Zone (m)",
                                        yaxis_title="Tax Rate (%)"
                                    )
                                    
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col2:
                        # Create heatmap of distance correlations
                        correlation_data = plot_data[['Tax Rate (%)', 'Tax Amount (SAR)'] + zone_distance_cols].copy()
                        
                        # Rename columns for better display
                        correlation_data.columns = [
                            col if not col.startswith('Distance_to_') else 
                            f"{col.replace(f'Distance_to_{selected_zone_city}_', '')} Zone Distance" 
                            for col in correlation_data.columns
                        ]
                        
                        # Calculate correlation
                        corr_matrix = correlation_data.corr()
                        
                        # Plot heatmap
                        fig_heatmap = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            title=f"Correlation between Zone Distances and Tax Metrics ({selected_zone_city})"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No distance data available for analysis.")

            # Add a section to highlight potential optimization opportunities
            st.subheader("Optimization Opportunities")

            # Find plots that might benefit from strategic development
            if 'Zone' in filtered_data.columns and 'Land Status ( Developed / Undeveloped / Partially Developed/WIP)' in filtered_data.columns:
                # Look for undeveloped plots in high-tax zones
                high_tax_undeveloped = filtered_data[
                    (filtered_data['Zone'] == 'Red') & 
                    (filtered_data['Land Status ( Developed / Undeveloped / Partially Developed/WIP)'] == 'Undeveloped') &
                    (filtered_data['Exemption from WLT (Yes or No)'] != 'Yes')
                ]
                
                if not high_tax_undeveloped.empty:
                    st.write(f"Found {len(high_tax_undeveloped)} undeveloped plots in high-tax (Red) zones that could benefit from development or rezoning.")
                    
                    # Show top plots by tax amount
                    top_plots = high_tax_undeveloped.sort_values('Tax Amount (SAR)', ascending=False).head(10)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_top = px.bar(
                            top_plots,
                            x='Plot number',
                            y='Tax Amount (SAR)',
                            color='City',
                            title="Top Undeveloped Plots in High-Tax Zones by Tax Amount",
                            text_auto='.2s'
                        )
                        st.plotly_chart(fig_top, use_container_width=True)
                    
                    with col2:
                        # Estimate potential savings if partially developed
                        top_plots['Potential Savings (SAR)'] = top_plots['Tax Amount (SAR)'] * 0.5  # 50% reduction if partially developed
                        
                        fig_savings = px.bar(
                            top_plots,
                            x='Plot number',
                            y='Potential Savings (SAR)',
                            color='City',
                            title="Potential Annual Tax Savings with Partial Development",
                            text_auto='.2s'
                        )
                        st.plotly_chart(fig_savings, use_container_width=True)
                    
                    # Show the actual data table
                    st.write("Top plots for potential optimization:")
                    st.dataframe(
                        top_plots[[
                            'Plot number', 'City', 'Total land plot size (sqm)', 
                            'Current value of owned land (SAR) as per business plan', 
                            'Tax Amount (SAR)', 'Potential Savings (SAR)'
                        ]].reset_index(drop=True),
                        use_container_width=True
                    )
                else:
                    st.info("No undeveloped plots in high-tax zones found in the current data selection.")
            else:
                st.info("Zone or Land Status data not available for optimization analysis.")
    
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
# Remove commas and then convert to float
        data['Current value of owned land (SAR) as per business plan'] = data['Current value of owned land (SAR) as per business plan'].str.replace(',', '').astype(float)
        
        total_potential_tax = data['Current value of owned land (SAR) as per business plan'].sum() * 0.025
        data['Tax Amount (SAR)']=data['Tax Amount (SAR)'].astype(float)
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
            'Total land plot size (sqm)', 'Current value of owned land (SAR) as per business plan',
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