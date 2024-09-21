import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import configparser

# Load the config
config = configparser.ConfigParser()
config.read('config.ini')
output = config['SIMULATION']['output']

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv(output, parse_dates=['time'])
    return df

# Create the map with taxi paths
def create_map(data, time, selected_taxi=None):
    current_data = data[data['time'] <= time]
    m = folium.Map(location=[current_data['lat'].mean(), current_data['lon'].mean()], zoom_start=12)
    
    color_map = {
        'waiting': 'blue',
        'to_cluster': 'green',
        'to_passenger': 'orange',
        'to_destination': 'red'
    }
    
    # Group the data by taxi ID
    grouped = current_data.groupby('id')
    
    for taxi_id, group in grouped:
        if selected_taxi is not None and taxi_id != selected_taxi:
            continue
        
        # Sort the group by time to ensure correct path
        group = group.sort_values('time')
        
        # Create a path for each taxi
        path = group[['lat', 'lon']].values.tolist()
        if len(path) > 1:
            folium.PolyLine(
                path,
                weight=2,
                color='black',
                opacity=0.5
            ).add_to(m)
        
        # Add a marker for the current position
        last_position = group.iloc[-1]
        
        folium.Marker(
            location=[last_position['lat'], last_position['lon']],
            radius=4,
            popup=f"ID: {last_position['id']}, Status: {last_position['status']}",
            color=color_map.get(last_position['status']),
            icon=folium.Icon(color=color_map.get(last_position['status']), icon='car' if last_position['status']!= 'to_destination' else 'person', prefix='fa'),
            fill=True
        ).add_to(m)
    
    return m

# Streamlit app
def main():
    st.title("Taxi Visualization with Paths")
    
    df = load_data()
    
    st.sidebar.header("Controls")
    
    min_time = df['time'].min().to_pydatetime()
    max_time = df['time'].max().to_pydatetime()
    
    # Use session state to persist the selected time across reruns
    if 'selected_time' not in st.session_state:
        st.session_state.selected_time = min_time

    # Time slider
    selected_time = st.sidebar.slider(
        "Select Time",
        min_value=min_time,
        max_value=max_time,
        value=st.session_state.selected_time,
        format="YYYY-MM-DD HH:mm:ss",
        key="time_slider"
    )

    # Update session state
    st.session_state.selected_time = selected_time

    # Buttons for minute navigation
    col1, col2 = st.sidebar.columns(2)
    if col1.button("◀ Previous Minute"):
        new_time = selected_time - timedelta(minutes=1)
        if new_time >= min_time:
            st.session_state.selected_time = new_time

    if col2.button("Next Minute ▶"):
        new_time = selected_time + timedelta(minutes=1)
        if new_time <= max_time:
            st.session_state.selected_time = new_time

    # Convert selected_time back to Timestamp for filtering
    selected_time = pd.Timestamp(selected_time)
    
    # Taxi selection dropdown
    taxi_ids = sorted(df['id'].unique())
    selected_taxi = st.sidebar.selectbox(
        "Select Taxi (optional)",
        ["All Taxis"] + taxi_ids,
        index=0
    )
    
    # Create and display map
    selected_taxi_id = None if selected_taxi == "All Taxis" else selected_taxi
    m = create_map(df, selected_time, selected_taxi_id)
    folium_static(m)
    
    # Display current time and taxi counts
    st.write(f"Current Time: {selected_time}")
    current_data = df[df['time'] <= selected_time]
    for status in ['waiting', 'to_cluster', 'to_passenger', 'to_destination']:
        count = current_data[current_data['status'] == status].shape[0]
        st.write(f"Taxis {status}: {count}")

if __name__ == "__main__":
    main()