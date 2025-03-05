import pandas as pd
import gdelt
from datetime import datetime, timedelta
import time
import os

# Initialize GDELT object
gd = gdelt.gdelt(version=2)

# Function to fetch hourly GDELT Events Data
def fetch_hourly_gdelt_events(start_date, end_date):
    """Fetches GDELT Events Data hourly"""
    
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_events = []
    
    while current_date <= end_date_dt:
        for hour in range(24):
            start_hour = current_date.replace(hour=hour, minute=0, second=0)
            end_hour = start_hour + timedelta(hours=1)
            
            start_str = start_hour.strftime('%Y-%m-%dT%H:%M:%S')
            end_str = end_hour.strftime('%Y-%m-%dT%H:%M:%S')
            
            print(f"Fetching GDELT Events for {start_str} - {end_str}")
            
            try:
                results = gd.Search([start_str, end_str], table='events', output='df')
                df = pd.DataFrame(results)
                
                if not df.empty:
                    df['DATE'] = pd.to_datetime(df['SQLDATE'].astype(str), format='%Y%m%d')
                    df['Hour'] = df['DATE'] + pd.to_timedelta(df['ActionGeo_Lat'] % 24, unit='h')
                    df.set_index('Hour', inplace=True)
                    all_events.append(df)
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
        
        current_date += timedelta(days=1)
    
    return pd.concat(all_events) if all_events else pd.DataFrame()

# Function to categorize and estimate attendance
def categorize_and_estimate_attendance(events_df):
    """Categorizes events and estimates attendance based on mentions and sentiment."""
    
    events_df['EventCategory'] = 'Other'
    events_df['Estimated_Attendance'] = events_df['NumMentions'].fillna(0) * (1 + (events_df['AvgTone'] / 100))
    
    # Categorizing events based on CAMEO codes
    events_df.loc[events_df['EventCode'].astype(str).str.startswith('16'), 'EventCategory'] = 'Public Gathering'
    events_df.loc[events_df['EventCode'].astype(str).str.startswith('14'), 'EventCategory'] = 'Protest & Demonstration'
    events_df.loc[events_df['EventCode'].astype(str).str.startswith('17'), 'EventCategory'] = 'Sports & Competition'
    events_df.loc[events_df['EventCode'].astype(str).str.startswith('03') | events_df['EventCode'].astype(str).str.startswith('04'), 'EventCategory'] = 'Religious Event'
    events_df.loc[events_df['EventCode'].astype(str).str.startswith('05') | events_df['EventCode'].astype(str).str.startswith('06'), 'EventCategory'] = 'Cultural & Tourism Event'
    
    return events_df[['EventCategory', 'Estimated_Attendance']].reset_index()

# Main Execution
if __name__ == "__main__":
    start_date = '2021-01-01'
    end_date = '2021-01-02'
    
    print("Fetching hourly GDELT events data...")
    events_df = fetch_hourly_gdelt_events(start_date, end_date)
    
    if not events_df.empty:
        print("Categorizing events and estimating attendance...")
        final_df = categorize_and_estimate_attendance(events_df)
        
        # Save to CSV
        if not os.path.exists("data"):
            os.makedirs("data")
            final_df.to_csv("data/hourly_event_attendance.csv", index=False)
        print("Saved hourly event dataset: hourly_event_attendance.csv")
    else:
        print("No event data found for the given date range.")
