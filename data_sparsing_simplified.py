import pandas as pd
import gdelt
import os
from datetime import datetime, timedelta
import time
import sys
import traceback

# Initialize GDELT object (version 2 for 15-minute updates)
print("Initializing GDELT API...")
try:
    gd = gdelt.gdelt(version=2)
    print("GDELT API initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize GDELT API: {e}")
    print("Please make sure the gdelt package is installed (pip install gdeltdoc)")
    sys.exit(1)

# Define time period - shorter period for testing
print("Setting up time period...")
start_date = '2021-01-01'
end_date = '2021-01-10'  # Reduced for testing
print(f"Time period: {start_date} to {end_date}")

# Define consistent directory structure
BASE_DIR = 'simplified_data'
GKG_DIR = os.path.join(BASE_DIR, 'gkg')
EVENTS_DIR = os.path.join(BASE_DIR, 'events')
WEATHER_DIR = os.path.join(BASE_DIR, 'weather_energy')
FINAL_DIR = os.path.join(BASE_DIR, 'final')

# Create all required directories
print("Creating directory structure...")
for directory in [GKG_DIR, EVENTS_DIR, WEATHER_DIR, FINAL_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"- Directory created/verified: {directory}")

def fetch_gdelt_data(data_type, start_date, end_date, chunk_size_days=5):  # Smaller chunk size
    """Fetch GDELT data (GKG or events) for Delhi in manageable chunks"""
    print(f"\n{'=' * 50}")
    print(f"FETCHING {data_type.upper()} DATA FOR DELHI")
    print(f"{'=' * 50}")
    print(f"Time range: {start_date} to {end_date}")
    
    all_data = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    start_time = time.time()
    chunks_processed = 0
    chunks_with_data = 0
    
    while current_date < end_date_dt:
        chunk_end_date = min(current_date + timedelta(days=chunk_size_days), end_date_dt)
        date_str = current_date.strftime('%Y-%m-%d')
        end_str = chunk_end_date.strftime('%Y-%m-%d')
        
        print(f"Processing chunk {chunks_processed+1}: {date_str} to {end_str}...")
        try:
            # Add timeout to prevent hanging
            results = gd.Search([date_str, end_str], 
                             table=data_type, 
                             output='df',
                             coverage=False)  # Remove unsupported parameter
            
            # Check if results are valid
            if results is None:
                print(f"  - No results returned for this chunk")
                chunks_processed += 1
                current_date = chunk_end_date
                continue
                
            # Convert to DataFrame
            print(f"  - Converting results to DataFrame...")
            df = pd.DataFrame(results)
            chunks_processed += 1
            
            if len(df) == 0:
                print(f"  - No data found for this chunk")
                current_date = chunk_end_date
                continue
                
            print(f"  - Found {len(df)} records before filtering")
            
            # Filter for Delhi
            location_col = 'Locations' if data_type == 'gkg' else 'ActionGeo_FullName'
            if location_col in df.columns:
                print(f"  - Filtering for Delhi in {location_col}...")
                df = df[df[location_col].str.contains('Delhi', case=False, na=False)]
                print(f"  - {len(df)} records match Delhi filter")
            else:
                print(f"  - Warning: Location column '{location_col}' not found")
            
            # Process date columns
            print(f"  - Processing date columns...")
            if data_type == 'gkg' and 'DATE' in df.columns and len(df) > 0:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H%M%S', errors='coerce')
                df = df.dropna(subset=['DATE'])
                df.set_index('DATE', inplace=True)
                print(f"  - Set index to DATE column")
            elif data_type == 'events' and 'SQLDATE' in df.columns and len(df) > 0:
                # Create datetime from SQLDATE
                df['DATE'] = pd.to_datetime(df['SQLDATE'].astype(str), format='%Y%m%d', errors='coerce')
                # Add time information if available
                if 'DATEADDED' in df.columns:
                    try:
                        time_part = pd.to_datetime(df['DATEADDED'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
                        df['DATE'] = df.apply(
                            lambda x: pd.Timestamp.combine(
                                x['DATE'].date(), 
                                time_part.loc[x.name].time() if pd.notna(time_part.loc[x.name]) else pd.Timestamp('00:00:00').time()
                            ) if pd.notna(x['DATE']) else pd.NaT,
                            axis=1
                        )
                    except Exception as e:
                        print(f"  - Error processing time part: {e}")
                
                df = df.dropna(subset=['DATE'])
                df.set_index('DATE', inplace=True)
                print(f"  - Set index to DATE column")
            
            if len(df) > 0:
                all_data.append(df)
                print(f"  - Added {len(df)} records to dataset")
                chunks_with_data += 1
            else:
                print(f"  - No data left after processing")
                
        except Exception as e:
            print(f"  - ERROR processing chunk {date_str} to {end_str}: {e}")
            print(f"  - {traceback.format_exc()}")
        
        # Move to next chunk
        current_date = chunk_end_date
        
        # Add a small delay to avoid API rate limits
        time.sleep(0.5)
    
    # Combine all data
    if all_data:
        try:
            print(f"\nCombining {len(all_data)} chunks of {data_type} data...")
            combined_df = pd.concat(all_data)
            processing_time = time.time() - start_time
            print(f"Total {data_type} records: {len(combined_df)}")
            if isinstance(combined_df.index, pd.DatetimeIndex) and len(combined_df) > 0:
                print(f"Time range: {combined_df.index.min()} to {combined_df.index.max()}")
            print(f"Processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
            print(f"Chunks processed: {chunks_processed} (with data: {chunks_with_data})")
            
            # Save raw data
            output_dir = GKG_DIR if data_type == 'gkg' else EVENTS_DIR
            raw_path = os.path.join(output_dir, f'{data_type}_raw.csv')
            combined_df.to_csv(raw_path)
            print(f"Raw data saved to: {raw_path}")
            
            # Create data info file
            create_data_info_file(combined_df, data_type, output_dir, 'raw')
            
            return combined_df
        except Exception as e:
            print(f"ERROR combining {data_type} data: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()
    else:
        print(f"No {data_type} data found")
        
        # Create empty file to indicate we tried
        output_dir = GKG_DIR if data_type == 'gkg' else EVENTS_DIR
        empty_path = os.path.join(output_dir, f'{data_type}_empty.txt')
        with open(empty_path, 'w') as f:
            f.write(f"No {data_type} data found for period {start_date} to {end_date}\n")
        print(f"Created empty indicator file: {empty_path}")
        
        return pd.DataFrame()

def create_data_info_file(df, data_type, output_dir, stage):
    """Create a comprehensive information file about the dataset"""
    if len(df) == 0:
        return
    
    info_path = os.path.join(output_dir, f'{data_type}_{stage}_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"{data_type.upper()} DATA INFORMATION - {stage.upper()} STAGE\n")
        f.write(f"{'=' * 50}\n\n")
        
        # Basic stats
        f.write(f"Records: {len(df)}\n")
        if isinstance(df.index, pd.DatetimeIndex):
            f.write(f"Time range: {df.index.min()} to {df.index.max()}\n")
        
        # Time granularity if applicable
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            time_diffs = df.index.to_series().diff().value_counts()
            most_common_diff = time_diffs.index[0] if len(time_diffs) > 0 else pd.Timedelta('0 days')
            f.write(f"Most common time difference: {most_common_diff}\n")
            
            # Top 3 most common time differences
            f.write("\nTime difference distribution:\n")
            for diff, count in time_diffs.head(3).items():
                percentage = (count / len(df.index) - 1) * 100 if len(df.index) > 1 else 0
                f.write(f"- {diff}: {count} occurrences ({percentage:.1f}%)\n")
        
        # Column information
        f.write(f"\nColumns ({len(df.columns)}):\n")
        f.write(f"{'-' * 50}\n")
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            f.write(f"- {col} ({dtype}): {missing_pct:.1f}% missing\n")
            
            # For categorical columns, show value distribution
            if df[col].dtype == 'object' and df[col].nunique() < 10:
                f.write("  Values: " + ", ".join(str(v) for v in df[col].unique()[:5]) + "\n")
            # For numeric columns, show summary stats
            elif df[col].dtype in ['int64', 'float64']:
                f.write(f"  Range: {df[col].min()} to {df[col].max()}, Mean: {df[col].mean():.2f}\n")
    
    print(f"Data info saved to: {info_path}")

def process_gkg_data(gkg_df):
    """Process GKG data to extract themes and tone"""
    if len(gkg_df) == 0:
        return pd.DataFrame()
    
    print(f"\n{'=' * 50}")
    print(f"PROCESSING GKG DATA")
    print(f"{'=' * 50}")
        
    # Define event categories
    categories = {
        'Cultural_Event': ['CULTURE', 'FESTIVAL', 'ART', 'MUSEUM', 'HERITAGE'],
        'Entertainment_Event': ['ENTERTAINMENT', 'MOVIE', 'FILM', 'MUSIC', 'CONCERT', 'THEATER', 'THEATRE'],
        'Tourism_Event': ['TOURISM', 'TRAVEL', 'VACATION', 'DESTINATION'],
        'Sports_Event': ['SPORT', 'ATHLETIC', 'OLYMPIC', 'CHAMPIONSHIP', 'TOURNAMENT'],
        'Religious_Event': ['RELIGION', 'RELIGIOUS', 'SPIRITUAL', 'WORSHIP', 'CEREMONY'],
        'Food_Event': ['FOOD', 'CUISINE', 'CULINARY', 'RESTAURANT', 'DINING']
    }
    
    # Initialize category columns
    for category in categories.keys():
        gkg_df[category] = 0
    
    # Search for themes
    print("Extracting event categories from themes...")
    for category, keywords in categories.items():
        # Check 'Themes' column
        if 'Themes' in gkg_df.columns:
            mask = False
            for keyword in keywords:
                mask = mask | gkg_df['Themes'].astype(str).str.contains(keyword, case=False, na=False)
            gkg_df[category] = mask.astype(int)
            
        # Also check 'V2Themes' column
        if 'V2Themes' in gkg_df.columns:
            mask = False
            for keyword in keywords:
                mask = mask | gkg_df['V2Themes'].astype(str).str.contains(keyword, case=False, na=False)
            gkg_df[category] = gkg_df[category] | mask.astype(int)
    
    # Process V2Tone
    if 'V2Tone' in gkg_df.columns:
        print("Processing V2Tone sentiment data...")
        try:
            # Extract first value (average tone)
            gkg_df['V2Tone_avg'] = gkg_df['V2Tone'].str.split(',').str[0]
            gkg_df['V2Tone_avg'] = pd.to_numeric(gkg_df['V2Tone_avg'], errors='coerce')
            
            # Report on tone distribution
            print(f"Tone analysis: ")
            print(f"- Mean tone: {gkg_df['V2Tone_avg'].mean():.2f}")
            print(f"- Negative tones: {(gkg_df['V2Tone_avg'] < 0).sum()} articles")
            print(f"- Positive tones: {(gkg_df['V2Tone_avg'] > 0).sum()} articles")
            
            # Remove original column to save space
            gkg_df = gkg_df.drop('V2Tone', axis=1)
        except Exception as e:
            print(f"Error processing V2Tone: {e}")
            if 'V2Tone' in gkg_df.columns:
                gkg_df = gkg_df.drop('V2Tone', axis=1)
    
    # Report on found categories
    print("\nEvent categories in GKG data:")
    for category in categories.keys():
        count = gkg_df[category].sum()
        percentage = (count / len(gkg_df)) * 100 if len(gkg_df) > 0 else 0
        print(f"- {category}: {count} entries ({percentage:.1f}%)")
    
    # Save processed data
    processed_path = os.path.join(GKG_DIR, 'gkg_processed.csv')
    gkg_df.to_csv(processed_path)
    print(f"Processed GKG data saved to: {processed_path}")
    
    # Create info file
    create_data_info_file(gkg_df, 'gkg', GKG_DIR, 'processed')
    
    return gkg_df

def process_events_data(events_df):
    """Process Events data to categorize events by CAMEO codes"""
    if len(events_df) == 0:
        return pd.DataFrame()
    
    print(f"\n{'=' * 50}")
    print(f"PROCESSING EVENTS DATA")
    print(f"{'=' * 50}")
    
    # Create event category columns
    event_categories = {
        'Protests_Count': [r'^14'],
        'Demonstrations_Count': [r'^145'],
        'Political_Events_Count': [r'^03', r'^04', r'^05'],
        'Conflicts_Count': [r'^17', r'^18', r'^19', r'^20'],
        'Gatherings_Count': [r'^16']
    }
    
    # Initialize columns
    for category in event_categories.keys():
        events_df[category] = 0
    
    # Categorize events if EventCode exists
    if 'EventCode' in events_df.columns:
        print("Categorizing events by CAMEO codes...")
        for category, patterns in event_categories.items():
            mask = False
            for pattern in patterns:
                mask = mask | events_df['EventCode'].astype(str).str.match(pattern)
            events_df.loc[mask, category] = 1
    
    # Report on found categories
    print("\nEvent categories found:")
    for category in event_categories.keys():
        count = events_df[category].sum()
        percentage = (count / len(events_df)) * 100 if len(events_df) > 0 else 0
        print(f"- {category}: {count} events ({percentage:.1f}%)")
    
    # Save processed data
    processed_path = os.path.join(EVENTS_DIR, 'events_processed.csv')
    events_df.to_csv(processed_path)
    print(f"Processed Events data saved to: {processed_path}")
    
    # Create info file
    create_data_info_file(events_df, 'events', EVENTS_DIR, 'processed')
    
    return events_df

def import_power_data(file_path, start_date=None, end_date=None):
    """Import power and weather data"""
    print(f"\n{'=' * 50}")
    print(f"IMPORTING POWER & WEATHER DATA")
    print(f"{'=' * 50}")
    print(f"Source file: {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Process datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        else:
            # Try to construct datetime from components
            date_components = ['year', 'month', 'day', 'hour', 'minute']
            if all(col in df.columns for col in date_components):
                df['datetime'] = pd.to_datetime(df[date_components])
                df.set_index('datetime', inplace=True)
            else:
                # Assume first column might be date
                df.set_index(df.columns[0], inplace=True)
                df.index = pd.to_datetime(df.index, errors='coerce')
        
        # Filter by date if specified
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        # Identify energy load columns vs. weather columns
        energy_cols = [col for col in df.columns if any(term in col.lower() for term in ['power', 'demand', 'kw', 'load', 'energy'])]
        weather_cols = [col for col in df.columns if col.lower() in ['temp', 'dwpt', 'rhum', 'wspd', 'pres', 'wdir', 'temperature', 'humidity']]
        
        print(f"\nColumns identified:")
        print(f"Energy columns: {', '.join(energy_cols)}")
        print(f"Weather columns: {', '.join(weather_cols)}")
        
        print(f"Power data: {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        # Save raw data
        raw_path = os.path.join(WEATHER_DIR, 'weather_energy_raw.csv')
        df.to_csv(raw_path)
        print(f"Raw power and weather data saved to: {raw_path}")
        
        # Create info file
        create_data_info_file(df, 'weather_energy', WEATHER_DIR, 'raw')
        
        return df
        
    except Exception as e:
        print(f"Error importing power data: {e}")
        return pd.DataFrame()

def resample_to_15min(df, data_type, is_event_data=False):
    """Resample data to 15-minute intervals with appropriate handling"""
    if len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
        print(f"Cannot resample {data_type} data - empty or no datetime index")
        return pd.DataFrame()
    
    print(f"\n{'=' * 50}")
    print(f"RESAMPLING {data_type.upper()} TO 15-MINUTE INTERVALS")
    print(f"{'=' * 50}")
    
    # Determine original frequency
    time_diffs = df.index.to_series().diff().value_counts()
    most_common_diff = time_diffs.index[0] if len(time_diffs) > 0 else pd.Timedelta('1 day')
    print(f"Original frequency: approximately {most_common_diff}")
    
    # Set aggregation functions based on column types and data source
    agg_funcs = {}
    
    if data_type == 'gkg':
        # For GKG data: sum event indicators, mean for other metrics
        for col in df.columns:
            if any(term in col for term in ['Event']):
                agg_funcs[col] = 'sum'
            elif col == 'V2Tone_avg':
                agg_funcs[col] = 'mean'
            elif df[col].dtype in [int, float]:
                agg_funcs[col] = 'mean'
                
    elif data_type == 'events':
        # For Events data: sum all event counts
        for col in df.columns:
            if any(term in col for term in ['Count']):
                agg_funcs[col] = 'sum'
            elif df[col].dtype in [int, float]:
                agg_funcs[col] = 'mean'
                
    elif data_type == 'weather_energy':
        # For power demand: sum, max, min, std
        # For weather metrics: mean, std
        for col in df.columns:
            if any(term in col.lower() for term in ['power', 'demand', 'kw', 'load', 'energy']):
                agg_funcs[col] = ['sum', 'max', 'min', 'std']
            elif col.lower() in ['temp', 'dwpt', 'rhum', 'wspd', 'pres', 'wdir', 'temperature', 'humidity']:
                agg_funcs[col] = ['mean', 'std']
            elif df[col].dtype in [int, float]:
                agg_funcs[col] = 'mean'
    
    # Create full 15-minute range
    full_range = pd.date_range(
        start=df.index.min().floor('D'),
        end=df.index.max().ceil('D'),
        freq='15min'
    )
    
    print(f"Creating {len(full_range)} 15-minute intervals")
    
    # Handle based on data type and original frequency
    try:
        if is_event_data:
            # FIXING EVENTS DATA HANDLING
            print(f"Special handling for {data_type} event data...")
            
            # First, create a template dataframe with all 15-minute intervals
            template_df = pd.DataFrame(index=full_range)
            
            # Step 1: For original event data, first aggregate to 15-minute intervals
            df_agg = df.groupby(pd.Grouper(freq='15min')).agg(agg_funcs)
            
            # Step 2: Reindex to ensure all 15-minute intervals exist
            df_agg = df_agg.reindex(full_range)
            
            # Step 3: Fill missing values with zeros for event columns
            count_cols = [col for col in df_agg.columns if any(term in col for term in ['Count', 'Event'])]
            if count_cols:
                df_agg[count_cols] = df_agg[count_cols].fillna(0)
            
            # For non-count columns, handle them appropriately
            non_count_cols = [col for col in df_agg.columns if col not in count_cols]
            if non_count_cols:
                # For GDELT GKG event indicators (which are binary) fill with 0
                if data_type == 'gkg':
                    event_indicator_cols = [col for col in non_count_cols if any(x in col for x in ['Cultural', 'Entertainment', 'Tourism', 'Sports', 'Religious', 'Food'])]
                    if event_indicator_cols:
                        df_agg[event_indicator_cols] = df_agg[event_indicator_cols].fillna(0)
                        
                # For V2Tone or other numeric columns, forward fill
                v2tone_cols = [col for col in non_count_cols if 'Tone' in col]
                if v2tone_cols:
                    df_agg[v2tone_cols] = df_agg[v2tone_cols].fillna(method='ffill')
                    # Then fill remaining with the mean
                    for col in v2tone_cols:
                        if df_agg[col].isna().any():
                            mean_val = df_agg[col].mean()
                            df_agg[col] = df_agg[col].fillna(mean_val if not pd.isna(mean_val) else 0)
            
            # Step 4: If the data is aggregated (e.g., daily), spread events across intervals
            if most_common_diff > pd.Timedelta('15 minutes'):
                print(f"Spreading {data_type} event data across time periods...")
                
                # For daily data, copy events to all 15-minute intervals within the day
                if most_common_diff >= pd.Timedelta('1 day'):
                    # Get only events that have occurred (non-zero count)
                    events_occurred = df_agg[df_agg[count_cols].sum(axis=1) > 0]
                    
                    # For each day with events
                    for day, day_events in events_occurred.groupby(events_occurred.index.date):
                        # Find all 15-minute intervals for this day
                        day_start = pd.Timestamp(day)
                        day_end = day_start + pd.Timedelta(days=1)
                        day_intervals = pd.date_range(start=day_start, end=day_end, freq='15min')[:-1]
                        
                        # Copy the event count across all intervals for this day
                        event_value = day_events.iloc[0][count_cols].values
                        for col_idx, col in enumerate(count_cols):
                            # Distribute the event over intervals (e.g., 1 event = 1/96 per 15-min interval)
                            df_agg.loc[day_intervals, col] = event_value[col_idx] / len(day_intervals)
                
                # For hourly data, copy events to all 15-minute intervals within the hour
                elif most_common_diff >= pd.Timedelta('1 hour'):
                    # Get only events that have occurred (non-zero count)
                    events_occurred = df_agg[df_agg[count_cols].sum(axis=1) > 0]
                    
                    # For each hour with events
                    for hour, hour_events in events_occurred.groupby([events_occurred.index.date, events_occurred.index.hour]):
                        # Find all 15-minute intervals for this hour
                        hour_start = pd.Timestamp(f"{hour[0]} {hour[1]:02d}:00:00")
                        hour_end = hour_start + pd.Timedelta(hours=1)
                        hour_intervals = pd.date_range(start=hour_start, end=hour_end, freq='15min')[:-1]
                        
                        # Copy the event count across all intervals for this hour
                        event_value = hour_events.iloc[0][count_cols].values
                        for col_idx, col in enumerate(count_cols):
                            # Distribute the event over intervals (e.g., 1 event = 1/4 per 15-min interval)
                            df_agg.loc[hour_intervals, col] = event_value[col_idx] / len(hour_intervals)
            
            print(f"Event data resampling complete: {len(df_agg)} 15-minute intervals")
        
        else:
            # Regular (non-event) data handling remains the same...
            if most_common_diff <= pd.Timedelta('15 minutes'):
                # Already 15min or finer, use standard resampling
                print("Data already at 15min or finer granularity - using standard aggregation")
                df_agg = df.groupby(pd.Grouper(freq='15min')).agg(agg_funcs)
            elif most_common_diff <= pd.Timedelta('1 hour'):
                # Hourly data: forward fill within the hour (3 intervals of 15min)
                print("Hourly data detected - using forward fill for 15min intervals")
                if data_type == 'weather_energy':
                    # For weather/energy, use proper aggregation first
                    df_hourly = df.groupby(pd.Grouper(freq='h')).agg(agg_funcs)
                    # Then resample hourly to 15min with forward fill
                    df_agg = df_hourly.resample('15min').ffill(limit=3)
                else:
                    df_agg = df.resample('15min').ffill(limit=3)
            else:
                # Daily data: forward fill up to a day (96 intervals of 15min)
                print("Daily data detected - using forward fill for 15min intervals")
                if data_type == 'weather_energy':
                    # For weather/energy, use proper aggregation first
                    df_daily = df.groupby(pd.Grouper(freq='D')).agg(agg_funcs)
                    # Then resample daily to 15min with forward fill
                    df_agg = df_daily.resample('15min').ffill(limit=95)
                else:
                    df_agg = df.resample('15min').ffill(limit=95)
        
        # Flatten multi-level columns if needed
        if isinstance(df_agg.columns, pd.MultiIndex):
            df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
            
        # Save resampled data
        output_dir = GKG_DIR if data_type == 'gkg' else (EVENTS_DIR if data_type == 'events' else WEATHER_DIR)
        resampled_path = os.path.join(output_dir, f'{data_type}_15min.csv')
        df_agg.to_csv(resampled_path)
        print(f"Resampled {data_type} data ({len(df_agg)} records) saved to: {resampled_path}")
        
        # Create info file
        create_data_info_file(df_agg, data_type, output_dir, '15min')
            
        return df_agg
        
    except Exception as e:
        print(f"Error resampling {data_type} data to 15-min intervals: {e}")
        print(traceback.format_exc())  # Add traceback for debugging
        return df

def main():
    """Main execution function"""
    print(f"\n{'#' * 60}")
    print(f"# STARTING GDELT-ENERGY DATA INTEGRATION")
    print(f"{'#' * 60}")
    
    # For testing, use smaller datasets or sample files
    try:
        # Step 1: Try to download sample GDELT data (reducing time range if needed)
        print("\nStep 1: Downloading GDELT data (trying with reduced time range if needed)...")
        
        # Try with GKG data first
        gkg_df = fetch_gdelt_data('gkg', start_date, end_date)
        if len(gkg_df) == 0:
            print("Retrying with smaller date range...")
            shorter_end = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
            gkg_df = fetch_gdelt_data('gkg', start_date, shorter_end)
        
        # Try with events data
        events_df = fetch_gdelt_data('events', start_date, end_date)
        if len(events_df) == 0:
            print("Retrying with smaller date range...")
            shorter_end = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
            events_df = fetch_gdelt_data('events', start_date, shorter_end)
            
        # Step 2: If still no data, create sample data
        if len(gkg_df) == 0:
            print("\nNo GKG data found, creating sample data...")
            gkg_df = create_sample_gkg_data()
            
        if len(events_df) == 0:
            print("\nNo Events data found, creating sample data...")
            events_df = create_sample_events_data()
            
        # Step 3: Try to read weather_energy.csv, or create sample
        print("\nStep 3: Importing power and weather data...")
        if os.path.exists('weather_energy.csv'):
            power_df = import_power_data('weather_energy.csv')
        else:
            print("weather_energy.csv not found, creating sample data...")
            power_df = create_sample_power_data()
            
        # Process the data files
        print("\nStep 4: Processing datasets...")
        gkg_df = process_gkg_data(gkg_df)
        events_df = process_events_data(events_df)
        
        # Resample to 15-minute intervals
        print("\nStep 5: Resampling all datasets to 15-minute intervals...")
        gkg_15min = resample_to_15min(gkg_df, 'gkg', is_event_data=True)
        events_15min = resample_to_15min(events_df, 'events', is_event_data=True)
        power_15min = resample_to_15min(power_df, 'weather_energy', is_event_data=False)
        
        # Combine final dataset
        print("\nStep 6: Creating final combined dataset...")
        final_df = combine_datasets(gkg_15min, events_15min, power_15min)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nERROR in main execution: {e}")
        print(traceback.format_exc())
        print("\nExecution failed. Please check the error messages above.")

# Helper functions for sample data creation
def create_sample_gkg_data():
    """Create sample GKG data for testing"""
    print("Creating sample GKG data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='6h')  # 'h' instead of 'H'
    n = len(dates)  # Get exact length
    
    # Make sure all arrays are exactly the same length
    themes = ['CULTURE', 'ENTERTAINMENT', 'SPORTS', 'RELIGION', 'FOOD']
    v2themes = ['TAX_FOREIGN', 'TOURISM', 'CULTURE', 'SPORT']
    
    # Create lists of exactly the right length
    data = {
        'DATE': dates,
        'Themes': [(themes[i % len(themes)]) for i in range(n)],
        'V2Themes': [(v2themes[i % len(v2themes)]) for i in range(n)],
        'V2Tone': ['2.5,3.4,1.2,4.6,20.1,0.5,100'] * n
    }
    
    df = pd.DataFrame(data)
    df.set_index('DATE', inplace=True)
    
    # Save sample data
    sample_path = os.path.join(GKG_DIR, 'gkg_sample.csv')
    df.to_csv(sample_path)
    print(f"Sample GKG data saved to: {sample_path}")
    return df

def create_sample_events_data():
    """Create sample Events data for testing"""
    print("Creating sample Events data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='8h')
    data = {
        'DATE': dates,
        'EventCode': ['145', '036', '170', '190', '160'] * (len(dates)//5 + 1),
        'ActionGeo_FullName': ['Delhi, India'] * len(dates)
    }
    df = pd.DataFrame(data)
    df.set_index('DATE', inplace=True)
    
    # Save sample data
    sample_path = os.path.join(EVENTS_DIR, 'events_sample.csv')
    df.to_csv(sample_path)
    print(f"Sample Events data saved to: {sample_path}")
    return df

def create_sample_power_data():
    """Create sample power and weather data for testing"""
    print("Creating sample power and weather data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    data = {
        'datetime': dates,
        'power_demand': [1000 + i % 500 for i in range(len(dates))],
        'temp': [25 + (i % 10) for i in range(len(dates))],
        'rhum': [60 + (i % 20) for i in range(len(dates))],
        'wspd': [5 + (i % 8) for i in range(len(dates))]
    }
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    
    # Save sample data
    sample_path = os.path.join(WEATHER_DIR, 'weather_energy_sample.csv')
    df.to_csv(sample_path)
    print(f"Sample power data saved to: {sample_path}")
    return df

def combine_datasets(gkg_df, events_df, power_df):
    """Combine all datasets into final dataset"""
    print("\nCombining all datasets...")
    
    # First combine GDELT datasets
    if len(gkg_df) > 0 and len(events_df) > 0:
        gdelt_combined = pd.merge(events_df, gkg_df, left_index=True, right_index=True,
                                  how='outer', suffixes=('_evt', '_gkg'))
        print(f"Combined GDELT dataset: {len(gdelt_combined)} records")
    else:
        gdelt_combined = gkg_df if len(gkg_df) > 0 else events_df
        print(f"Using only one GDELT source: {len(gdelt_combined)} records")
    
    # Merge with power data
    if len(gdelt_combined) > 0 and len(power_df) > 0:
        final_df = pd.merge(gdelt_combined, power_df, left_index=True, right_index=True, how='outer')
        print(f"Final dataset has {len(final_df)} records and {final_df.shape[1]} columns")
    else:
        final_df = gdelt_combined if len(gdelt_combined) > 0 else power_df
        print(f"Using only one data source for final dataset: {len(final_df)} records")
    
    # Save final dataset
    final_path = os.path.join(FINAL_DIR, 'finbal.csv')
    final_df.to_csv(final_path)
    print(f"Final dataset saved to: {final_path}")
    
    # Save info file
    create_data_info_file(final_df, 'final', FINAL_DIR, 'finbal')
    return final_df

# Run with main driver
if __name__ == "__main__":
    print("\nStarting data processing script...")
    main()
    print("\nScript execution completed.")