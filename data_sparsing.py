import pandas as pd
import gdelt
from datetime import datetime, timedelta
import time
import requests
from dateutil.relativedelta import relativedelta

# Initialize GDELT object (version 2 for 15-minute updates)
gd = gdelt.gdelt(version=2)

# Define time period (2021-2024)
start_date = '2021-01-01'  # Changed from 2020 to 2021
end_date = '2021-02-04'    # Changed to match the year

# Function to check data availability
def check_gdelt_availability(test_start_date, test_end_date, table='gkg'):
    """Test a small query to see if data is available for the given period"""
    try:
        print(f"Testing {table.upper()} data availability for {test_start_date} to {test_end_date}...")
        # Add timeout to prevent hanging
        results = gd.Search([test_start_date, test_end_date], 
                          table=table,
                          output='df',
                          coverage=False)
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            print(f"Data available for {test_start_date} to {test_end_date}, found {len(df)} records")
            return True
        else:
            print(f"No data available for {test_start_date} to {test_end_date}")
            return False
    except Exception as e:
        print(f"Error checking data for {test_start_date} to {test_end_date}: {e}")
        return False

# Function to fetch GKG data in robust manner
def fetch_delhi_gkg_data_robust(overall_start_date, overall_end_date, chunk_size_days=30):
    """Fetch GKG data in smaller chunks with retry mechanism"""
    all_gkg = []
    total_gkg_count = 0
    chunks_processed = 0
    chunks_with_data = 0
    
    # Convert string dates to datetime objects
    current_date = datetime.strptime(overall_start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(overall_end_date, '%Y-%m-%d')
    total_days = (end_date_dt - current_date).days
    days_processed = 0
    
    print(f"Starting GKG data collection for {total_days} days ({overall_start_date} to {overall_end_date})")
    start_time = time.time()
    
    while current_date < end_date_dt:
        # Calculate end of current chunk
        chunk_end_date = min(current_date + timedelta(days=chunk_size_days), end_date_dt)
        
        # Format dates for GDELT
        current_str = current_date.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end_date.strftime('%Y-%m-%d')
        chunks_processed += 1
        days_processed += (chunk_end_date - current_date).days
        
        # Progress indicator
        progress = days_processed / total_days * 100
        elapsed_time = time.time() - start_time
        print(f"[{progress:.1f}%] Processing chunk {chunks_processed}: {current_str} to {chunk_end_str} | Elapsed: {elapsed_time:.1f}s")
        
        # Check if data is available for this period
        if check_gdelt_availability(current_str, chunk_end_str, 'gkg'):
            # Try to fetch data with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Fetching GKG data for {current_str} to {chunk_end_str} (Attempt {attempt+1}/{max_retries})")
                    
                    # Basic search with minimal parameters
                    results = gd.Search([current_str, chunk_end_str], 
                                      table='gkg',
                                      output='df')
                    
                    # Convert to DataFrame
                    df_gkg = pd.DataFrame(results)
                    
                    # Filter for Delhi after receiving all results
                    if 'Locations' in df_gkg.columns:
                        df_gkg = df_gkg[df_gkg['Locations'].str.contains('Delhi', na=False)]
                    
                    # Convert DATE to datetime
                    if 'DATE' in df_gkg.columns and len(df_gkg) > 0:
                        df_gkg['DATE'] = pd.to_datetime(df_gkg['DATE'], format='%Y%m%d%H%M%S')
                        df_gkg.set_index('DATE', inplace=True)
                    
                    # Add to our collection
                    if len(df_gkg) > 0:
                        all_gkg.append(df_gkg)
                        chunks_with_data += 1
                        total_gkg_count += len(df_gkg)
                        print(f"✓ Fetched {len(df_gkg)} Delhi GKG entries | Running total: {total_gkg_count}")
                    
                    # Break the retry loop if successful
                    break
                    
                except Exception as e:
                    print(f"Error on attempt {attempt+1}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)  # Exponential backoff
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to fetch data after {max_retries} attempts")
        
        # Move to next chunk
        current_date = chunk_end_date
    
    # Print summary statistics
    total_time = time.time() - start_time
    if all_gkg:
        combined_df = pd.concat(all_gkg)
        print(f"\nCollection complete:")
        print(f"- Total GKG entries: {total_gkg_count} across {chunks_with_data}/{chunks_processed} chunks")
        print(f"- Time range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"- Processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        return combined_df
    else:
        print("\nNo GKG data was collected after trying all date ranges!")
        return pd.DataFrame()

# Function to fetch Events data in robust manner
def fetch_delhi_events_robust(overall_start_date, overall_end_date, chunk_size_days=30):
    """Fetch Events data in smaller chunks with retry mechanism"""
    all_events = []
    total_events_count = 0
    chunks_processed = 0
    chunks_with_data = 0
    
    # Convert string dates to datetime objects
    current_date = datetime.strptime(overall_start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(overall_end_date, '%Y-%m-%d')
    total_days = (end_date_dt - current_date).days
    days_processed = 0
    
    print(f"Starting Events data collection for {total_days} days ({overall_start_date} to {overall_end_date})")
    start_time = time.time()
    
    while current_date < end_date_dt:
        # Calculate end of current chunk
        chunk_end_date = min(current_date + timedelta(days=chunk_size_days), end_date_dt)
        
        # Format dates for GDELT
        current_str = current_date.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end_date.strftime('%Y-%m-%d')
        chunks_processed += 1
        days_processed += (chunk_end_date - current_date).days
        
        # Progress indicator
        progress = days_processed / total_days * 100
        elapsed_time = time.time() - start_time
        print(f"[{progress:.1f}%] Processing chunk {chunks_processed}: {current_str} to {chunk_end_str} | Elapsed: {elapsed_time:.1f}s")
        
        # Check if data is available for this period
        if check_gdelt_availability(current_str, chunk_end_str, 'events'):
            # Try to fetch data with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Fetching Events data for {current_str} to {chunk_end_str} (Attempt {attempt+1}/{max_retries})")
                    
                    # Basic search with minimal parameters
                    results = gd.Search([current_str, chunk_end_str], 
                                      table='events',
                                      output='df')
                    
                    # Convert to DataFrame
                    df_events = pd.DataFrame(results)
                    
                    # Filter for Delhi after receiving all results
                    if 'ActionGeo_FullName' in df_events.columns:
                        df_events = df_events[df_events['ActionGeo_FullName'].str.contains('Delhi', na=False)]
                    
                    # Create a proper datetime index
                    if len(df_events) > 0:
                        # Try to create a datetime index from SQLDATE and add time information
                        if 'SQLDATE' in df_events.columns:
                            try:
                                # First validate SQLDATE format - should be 8 digits (YYYYMMDD)
                                valid_dates_mask = df_events['SQLDATE'].astype(str).str.match(r'^\d{8}$')
                                if not all(valid_dates_mask):
                                    print(f"DATA WARNING: Found {(~valid_dates_mask).sum()} records with invalid SQLDATE format")
                                    df_events = df_events[valid_dates_mask].copy()
                                    
                                # Convert SQLDATE to datetime and validate year range
                                date_part = pd.to_datetime(df_events['SQLDATE'].astype(str), format='%Y%m%d', errors='coerce')
                                valid_year_mask = (date_part.dt.year >= 2000) & (date_part.dt.year <= datetime.now().year)
                                
                                if not all(valid_year_mask):
                                    print(f"DATA WARNING: Found {(~valid_year_mask).sum()} records outside valid year range (2000-present)")
                                    df_events = df_events[valid_year_mask].copy()
                                    date_part = date_part[valid_year_mask]
                                
                                # Print date statistics for verification
                                if len(date_part) > 0:
                                    print(f"DATE VALIDATION: Years in data range from {date_part.dt.year.min()} to {date_part.dt.year.max()}")
                                    
                                # Use DATEADDED for time information if available
                                if 'DATEADDED' in df_events.columns:
                                    try:
                                        time_part = pd.to_datetime(df_events['DATEADDED'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
                                        df_events['DATE'] = date_part.dt.date.astype(str) + ' ' + time_part.dt.time.astype(str)
                                        df_events['DATE'] = pd.to_datetime(df_events['DATE'], errors='coerce')
                                    except Exception as e:
                                        print(f"Error extracting time from DATEADDED: {e}")
                                        # Default to noon if time extraction fails
                                        df_events['DATE'] = date_part
                                else:
                                    # No time information, use noon as default
                                    df_events['DATE'] = date_part
                                
                                # Set index
                                df_events.set_index('DATE', inplace=True)
                                print(f"Created datetime index for events data with range: {df_events.index.min()} to {df_events.index.max()}")
                            except Exception as e:
                                print(f"ERROR: Failed to process date information: {e}")
                                # If date processing fails, create a simple date column
                                df_events['DATE'] = pd.to_datetime(current_str) + pd.Timedelta(hours=12)
                                df_events.set_index('DATE', inplace=True)
                    
                    # Add to our collection
                    if len(df_events) > 0:
                        all_events.append(df_events)
                        chunks_with_data += 1
                        total_events_count += len(df_events)
                        print(f"✓ Fetched {len(df_events)} Delhi Events entries | Running total: {total_events_count}")
                    
                    # Break the retry loop if successful
                    break
                    
                except Exception as e:
                    print(f"Error on attempt {attempt+1}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)  # Exponential backoff
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to fetch data after {max_retries} attempts")
        
        # Move to next chunk
        current_date = chunk_end_date
    
    # Print summary statistics
    total_time = time.time() - start_time
    if all_events:
        combined_df = pd.concat(all_events)
        print(f"\nCollection complete:")
        print(f"- Total Events entries: {total_events_count} across {chunks_with_data}/{chunks_processed} chunks")
        print(f"- Time range: {combined_df.index.min()} to {combined_df.index.max() if isinstance(combined_df.index, pd.DatetimeIndex) else 'Not a datetime index'}")
        print(f"- Processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Print index type information for debugging
        print(f"- Index type: {type(combined_df.index)}")
        return combined_df
    else:
        print("\nNo Events data was collected after trying all date ranges!")
        return pd.DataFrame()

# Function to categorize events by CAMEO codes
def categorize_events(df_events):
    """Add category columns based on CAMEO event codes"""
    # Create event category columns
    df_events['Protests_Count'] = 0
    df_events['Demonstrations_Count'] = 0
    df_events['Political_Events_Count'] = 0
    df_events['Conflicts_Count'] = 0 
    df_events['Gatherings_Count'] = 0
    
    if 'EventCode' in df_events.columns:
        # Protests (CAMEO code 14*)
        protest_mask = df_events['EventCode'].astype(str).str.startswith('14')
        df_events.loc[protest_mask, 'Protests_Count'] = 1
        
        # Demonstrations (subset of protest events, typically 145*)
        demo_mask = df_events['EventCode'].astype(str).str.startswith('145')
        df_events.loc[demo_mask, 'Demonstrations_Count'] = 1
        
        # Political events (CAMEO codes 03*, 04*, 05*)
        political_mask = (df_events['EventCode'].astype(str).str.startswith('03') | 
                         df_events['EventCode'].astype(str).str.startswith('04') | 
                         df_events['EventCode'].astype(str).str.startswith('05'))
        df_events.loc[political_mask, 'Political_Events_Count'] = 1
        
        # Conflicts (CAMEO codes 17*, 18*, 19*, 20*)
        conflict_mask = (df_events['EventCode'].astype(str).str.startswith('17') | 
                        df_events['EventCode'].astype(str).str.startswith('18') | 
                        df_events['EventCode'].astype(str).str.startswith('19') | 
                        df_events['EventCode'].astype(str).str.startswith('20'))
        df_events.loc[conflict_mask, 'Conflicts_Count'] = 1
        
        # Gatherings (CAMEO codes 16*)
        gathering_mask = df_events['EventCode'].astype(str).str.startswith('16')
        df_events.loc[gathering_mask, 'Gatherings_Count'] = 1
    
    # Print summary of categorized events
    print("DATA ANALYSIS: Event categories from CAMEO codes:")
    for category in ['Protests_Count', 'Demonstrations_Count', 'Political_Events_Count', 
                    'Conflicts_Count', 'Gatherings_Count']:
        count = df_events[category].sum()
        print(f"- {category}: {count} events ({count/len(df_events)*100:.1f}% of total)")
    
    return df_events

# Function to filter GKG data for cultural and entertainment events
def filter_cultural_events(df_gkg):
    """Filter and classify GKG data for cultural events and other categories"""
    # Initialize columns if they don't exist
    df_gkg['Cultural_Event'] = 0
    df_gkg['Entertainment_Event'] = 0  
    df_gkg['Tourism_Event'] = 0
    df_gkg['Sports_Event'] = 0
    df_gkg['Religious_Event'] = 0
    df_gkg['Food_Event'] = 0
    
    # Define theme keywords for each category
    theme_categories = {
        'Cultural_Event': ['CULTURE', 'FESTIVAL', 'ART', 'MUSEUM', 'HERITAGE'],
        'Entertainment_Event': ['ENTERTAINMENT', 'MOVIE', 'FILM', 'MUSIC', 'CONCERT', 'THEATER', 'THEATRE'],
        'Tourism_Event': ['TOURISM', 'TRAVEL', 'VACATION', 'DESTINATION'],
        'Sports_Event': ['SPORT', 'ATHLETIC', 'OLYMPIC', 'CHAMPIONSHIP', 'TOURNAMENT'],
        'Religious_Event': ['RELIGION', 'RELIGIOUS', 'SPIRITUAL', 'WORSHIP', 'CEREMONY'],
        'Food_Event': ['FOOD', 'CUISINE', 'CULINARY', 'RESTAURANT', 'DINING']
    }
    
    # Filter for various themes if the column exists
    if 'Themes' in df_gkg.columns:
        # Check each category
        for category, keywords in theme_categories.items():
            # Create a composite mask with OR logic
            mask = df_gkg['Themes'].astype(str).str.contains(keywords[0], case=False, na=False)
            for keyword in keywords[1:]:
                mask = mask | df_gkg['Themes'].astype(str).str.contains(keyword, case=False, na=False)
            
            # Set the category column
            df_gkg[category] = mask.astype(int)
    
    # Also check V2Themes which may contain more detailed data
    if 'V2Themes' in df_gkg.columns:
        # Similar logic but for V2Themes
        for category, keywords in theme_categories.items():
            for keyword in keywords:
                mask = df_gkg['V2Themes'].astype(str).str.contains(keyword, case=False, na=False)
                df_gkg[category] = df_gkg[category] | mask.astype(int)
    
    print(f"DATA ANALYSIS: Found event categories in GKG data:")
    for category in theme_categories.keys():
        count = df_gkg[category].sum()
        print(f"- {category}: {count} entries ({count/len(df_gkg)*100:.1f}%)")
    
    return df_gkg

# Function to clean V2Tone data
def clean_v2tone(df_gkg):
    """Extract and clean V2Tone data from GKG records"""
    if 'V2Tone' not in df_gkg.columns:
        print("V2Tone column not found in data")
        return df_gkg
        
    # Print diagnostic info about data types
    print(f"DATA INSPECTION: First 5 V2Tone values (type: {type(df_gkg['V2Tone'].iloc[0]) if len(df_gkg) > 0 else 'unknown'}):")
    print(df_gkg['V2Tone'].head().tolist())
    
    # Extract first value which is the average tone
    try:
        # First check if we're dealing with a comma-separated list
        if df_gkg['V2Tone'].iloc[0].find(',') > 0:
            print("V2Tone contains comma-separated values - extracting first value (avg tone)")
            # Extract the first value (avg tone) from comma-separated list
            df_gkg['V2Tone_avg'] = df_gkg['V2Tone'].str.split(',').str[0]
            
            # Convert to numeric
            df_gkg['V2Tone_avg'] = pd.to_numeric(df_gkg['V2Tone_avg'], errors='coerce')
            
            # Record stats
            nan_count = df_gkg['V2Tone_avg'].isna().sum()
            nan_percentage = (nan_count / len(df_gkg)) * 100 if len(df_gkg) > 0 else 0
            print(f"DATA CONVERSION: V2Tone_avg created. NaN count: {nan_count} ({nan_percentage:.1f}% of data)")
            
            # Drop original V2Tone
            df_gkg = df_gkg.drop('V2Tone', axis=1)
        else:
            # Try direct conversion if it's already a simple value
            df_gkg['V2Tone'] = pd.to_numeric(df_gkg['V2Tone'], errors='coerce')
            df_gkg.rename(columns={'V2Tone': 'V2Tone_avg'}, inplace=True)
    except Exception as e:
        print(f"DATA ERROR: Failed to process V2Tone column: {e}")
        # Remove problematic column if conversion fails
        if 'V2Tone' in df_gkg.columns:
            df_gkg = df_gkg.drop('V2Tone', axis=1)
    
    return df_gkg

# Function to aggregate data by time interval
def aggregate_by_time(df, freq='D', value_columns=None, agg_funcs=None):
    """
    Aggregate data by specified time interval (daily, hourly)
    
    Args:
        df: DataFrame with datetime index
        freq: Time frequency ('D' for daily, 'H' for hourly, etc)
        value_columns: List of columns to aggregate
        agg_funcs: Dict of column:aggregation_function
    
    Returns:
        Aggregated DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"ERROR: Cannot aggregate data without a datetime index. Current index type: {type(df.index)}")
        return df
    
    print(f"DATA AGGREGATION: Aggregating data to {freq} frequency")
    
    # Default aggregation: sum for count columns, mean for other numeric
    if agg_funcs is None:
        agg_funcs = {}
        # Sum for columns containing 'Count' or 'Event'
        for col in df.columns:
            if 'Count' in col or 'Event' in col:
                agg_funcs[col] = 'sum'
            elif df[col].dtype in [int, float]:
                agg_funcs[col] = 'mean'
    
    try:
        # Group by time period and aggregate
        df_agg = df.groupby(pd.Grouper(freq=freq)).agg(agg_funcs)
        
        # Fill NaN values with 0 for count columns
        count_cols = [col for col in df_agg.columns if 'Count' in col or 'Event' in col]
        if count_cols:
            df_agg[count_cols] = df_agg[count_cols].fillna(0)
            
        print(f"DATA SUCCESS: Aggregated data from {len(df)} rows to {len(df_agg)} time periods")
        return df_agg
    
    except Exception as e:
        print(f"DATA ERROR: Failed to aggregate by time: {e}")
        return df

# Function to import power demand and weather data
def import_power_weather_data(file_path, start_date=None, end_date=None):
    """
    Import power demand and weather data from CSV file
    
    Args:
        file_path: Path to the CSV file
        start_date: Optional start date to filter data (YYYY-MM-DD format)
        end_date: Optional end date to filter data (YYYY-MM-DD format)
    
    Returns:
        DataFrame with datetime index
    """
    print("\n" + "="*80)
    print("IMPORTING POWER DEMAND & WEATHER DATA")
    print("="*80)
    
    try:
        print(f"Reading data from {file_path}")
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if we have the datetime column
        if 'datetime' in df.columns:
            # Convert to datetime and set as index
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            print(f"Datetime column found and set as index")
        else:
            # Try to construct datetime from components if they exist
            date_components = ['year', 'month', 'day', 'hour', 'minute']
            if all(col in df.columns for col in date_components):
                print("Creating datetime index from date components")
                df['datetime'] = pd.to_datetime(
                    df[['year', 'month', 'day', 'hour', 'minute']].rename(
                        columns={'month': 'M', 'day': 'D', 'hour': 'h', 'minute': 'm'}
                    )
                )
                df.set_index('datetime', inplace=True)
            else:
                print("WARNING: No datetime column found. First column will be used as index.")
                # Assume the first column might be a date
                df.set_index(df.columns[0], inplace=True)
                # Try to convert the index to datetime
                df.index = pd.to_datetime(df.index, errors='coerce')
        
        # Filter by date range if specified
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            df = df[mask]
            print(f"Filtered data to range: {start_date} to {end_date}")
        
        # Check for power demand column - update to match actual column name with space
        power_cols = [col for col in df.columns if 'power' in col.lower() or 'demand' in col.lower()]
        if power_cols:
            print(f"Found power demand column(s): {', '.join(power_cols)}")
        else:
            print("WARNING: No obvious power demand column found. Please verify the data.")
        
        # Check for weather columns
        weather_cols = ['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']
        found_weather_cols = [col for col in weather_cols if col in df.columns]
        if found_weather_cols:
            print(f"Found weather columns: {', '.join(found_weather_cols)}")
        else:
            print("WARNING: No standard weather columns found. Please verify the data.")
        
        # Print data summary
        print(f"Successfully imported data with {len(df)} records")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Data frequency: {df.index.to_series().diff().median()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR importing power demand data: {e}")
        return pd.DataFrame()

# Function to combine GDELT data with power demand and weather data
def combine_gdelt_with_power_data(gdelt_data, power_weather_df, time_aggregation='D', output_dir='.'):
    """
    Combine GDELT data with power demand and weather data
    
    Args:
        gdelt_data: GDELT data (DataFrame or dict with 'gkg' and 'events' keys)
        power_weather_df: Power demand and weather data DataFrame
        time_aggregation: Time frequency for final dataset ('D' for daily, 'H' for hourly)
    
    Returns:
        Combined DataFrame
    """
    print("\n" + "="*80)
    print("COMBINING GDELT WITH POWER & WEATHER DATA")
    print("="*80)
    
    # Handle different formats of GDELT data
    if isinstance(gdelt_data, dict):
        # Extract data from dict
        gkg_df = gdelt_data.get('gkg', pd.DataFrame())
        events_df = gdelt_data.get('events', pd.DataFrame())
        
        # Use the GKG data as primary if available, otherwise use events
        gdelt_df = gkg_df if len(gkg_df) > 0 else events_df
    else:
        # Already a DataFrame (likely already aggregated)
        gdelt_df = gdelt_data
    
    # If either dataset is empty, return the other
    if len(gdelt_df) == 0:
        print("No GDELT data available. Returning power & weather data only.")
        return power_weather_df
    
    if len(power_weather_df) == 0:
        print("No power & weather data available. Returning GDELT data only.")
        return gdelt_df
    
    # Ensure both datasets have datetime indices
    if not isinstance(gdelt_df.index, pd.DatetimeIndex):
        print("ERROR: GDELT data does not have a datetime index.")
        return power_weather_df
    
    if not isinstance(power_weather_df.index, pd.DatetimeIndex):
        print("ERROR: Power & weather data does not have a datetime index.")
        return gdelt_df
    
    # Resample power & weather data to match the desired aggregation level
    print(f"Aggregating power & weather data to {time_aggregation} frequency")
    
    # Define aggregation functions for different columns
    # You may need to adjust these based on your specific columns
    agg_funcs = {}
    
    # Identify power demand columns
    power_cols = [col for col in power_weather_df.columns if 'power' in col.lower() or 'demand' in col.lower() or 'kw' in col.lower()]
    
    # Set aggregation functions for different column types
    for col in power_weather_df.columns:
        if col in power_cols:
            # For power demand: use sum instead of mean, plus keep max, min, std
            agg_funcs[col] = ['sum', 'max', 'min', 'std']
        elif col in ['temp', 'dwpt', 'rhum', 'wspd', 'pres']:
            # For weather metrics: use mean and std
            agg_funcs[col] = ['mean', 'std']
        elif power_weather_df[col].dtype in [int, float]:
            # For other numeric columns: use mean
            agg_funcs[col] = 'mean'
    
    # Perform the aggregation
    try:
        power_agg = power_weather_df.groupby(pd.Grouper(freq=time_aggregation)).agg(agg_funcs)
        
        # Flatten multi-level column names
        if isinstance(power_agg.columns, pd.MultiIndex):
            power_agg.columns = ['_'.join(col).strip() for col in power_agg.columns.values]
        
        print(f"Successfully aggregated power & weather data from {len(power_weather_df)} to {len(power_agg)} records")
    except Exception as e:
        print(f"ERROR aggregating power & weather data: {e}")
        return gdelt_df
    
    # Now merge the datasets on their indices
    try:
        print("Merging GDELT with power & weather data")
        combined_df = pd.merge(
            gdelt_df, 
            power_agg,
            left_index=True, 
            right_index=True,
            how='outer'
        )
        
        # Handle missing values appropriately
        # Fill NAs in event counts with 0
        event_cols = [col for col in combined_df.columns if 'Count' in col or 'Event' in col]
        if event_cols:
            combined_df[event_cols] = combined_df[event_cols].fillna(0)
        
        # For power and weather metrics, leave NAs as is or use forward fill
        # as they represent actual missing measurements
        
        print(f"Successfully merged datasets into {len(combined_df)} records with {combined_df.shape[1]} variables")
        
        # Save to CSV with new name "finbal.csv"
        output_path = os.path.join(output_dir, 'finbal.csv')
        combined_df.to_csv(output_path)
        print(f"Combined dataset saved to {output_path}")
        
        # And the combined file:
        combined_path = os.path.join(output_dir, f'gdelt_combined_{time_aggregation}.csv')
        combined_df.to_csv(combined_path)
        print(f"DATA EXPORT: Combined aggregated data saved to {combined_path}")
        
        return combined_df
    
    except Exception as e:
        print(f"ERROR merging datasets: {e}")
        return pd.DataFrame()

# Main execution function
def process_gdelt_data(time_aggregation='D', output_dir='.'):
    """
    Process GDELT data for Delhi
    
    Args:
        time_aggregation: Time frequency for aggregation
                        'D' = daily
                        'H' = hourly
                        'min' = minute-level (original 15-min granularity)
                        None = no aggregation
        output_dir: Directory to save output files
    """
    # Step 1: Process GKG Data
    print("\n" + "="*80)
    print("PROCESSING GKG DATA")
    print("="*80)
    gkg_df = fetch_delhi_gkg_data_robust(start_date, end_date)
    
    if len(gkg_df) == 0:
        print("DATA ALERT: No GKG data found.")
        gkg_df = pd.DataFrame()
    else:
        # Filter for cultural events and other categories
        print(f"DATA PROCESSING: Categorizing {len(gkg_df)} GKG entries by themes")
        gkg_df = filter_cultural_events(gkg_df)
        
        # Clean V2Tone data
        print("DATA CLEANING: Processing V2Tone data")
        gkg_df = clean_v2tone(gkg_df)
        
        # Save processed GKG data before aggregation
        gkg_out_path = os.path.join(output_dir, 'gkg_processed.csv')
        gkg_df.to_csv(gkg_out_path)
        print(f"DATA EXPORT: Processed GKG data ({len(gkg_df)} records) saved to {gkg_out_path}")
    
    # Step 2: Process Events Data
    print("\n" + "="*80)
    print("PROCESSING EVENTS DATA")
    print("="*80)
    events_df = fetch_delhi_events_robust(start_date, end_date)
    
    if len(events_df) == 0:
        print("DATA ALERT: No Events data found.")
        events_df = pd.DataFrame()
    else:
        # Categorize events
        print(f"DATA PROCESSING: Categorizing {len(events_df)} events by CAMEO codes")
        events_df = categorize_events(events_df)
        
        # Save processed events data before aggregation
        events_df.to_csv(f'{output_dir}/events_processed.csv')
        print(f"DATA EXPORT: Processed events data ({len(events_df)} records) saved to {output_dir}/events_processed.csv")
    
    # Step 3: Apply time-based aggregation if requested
    if time_aggregation and time_aggregation.lower() != 'none':
        print("\n" + "="*80)
        print(f"TIME AGGREGATION: {time_aggregation}")
        print("="*80)
        
        # Aggregate GKG data if available
        if len(gkg_df) > 0 and isinstance(gkg_df.index, pd.DatetimeIndex):
            gkg_agg = aggregate_by_time(gkg_df, freq=time_aggregation)
            gkg_agg_path = os.path.join(output_dir, f'gkg_aggregated_{time_aggregation}.csv')
            gkg_agg.to_csv(gkg_agg_path)
            print(f"DATA EXPORT: Aggregated GKG data saved to {gkg_agg_path}")
        else:
            gkg_agg = gkg_df
        
        # Aggregate Events data if available
        if len(events_df) > 0 and isinstance(events_df.index, pd.DatetimeIndex):
            events_agg = aggregate_by_time(events_df, freq=time_aggregation)
            events_agg_path = os.path.join(output_dir, f'events_aggregated_{time_aggregation}.csv')
            events_agg.to_csv(events_agg_path)
            print(f"DATA EXPORT: Aggregated Events data saved to {events_agg_path}")
        else:
            events_agg = events_df
        
        # Try to merge aggregated data
        if len(gkg_agg) > 0 and len(events_agg) > 0:
            try:
                # Both datasets have data, attempt to merge
                combined_df = pd.merge(events_agg, gkg_agg, 
                                      left_index=True, right_index=True, 
                                      how='outer', 
                                      suffixes=('_evt', '_gkg'))
                
                # Fill NaN values with 0 for count columns
                count_cols = [col for col in combined_df.columns if 'Count' in col or 'Event' in col]
                combined_df[count_cols] = combined_df[count_cols].fillna(0)
                
                combined_path = os.path.join(output_dir, f'gdelt_combined_{time_aggregation}.csv')
                combined_df.to_csv(combined_path)
                print(f"DATA EXPORT: Combined aggregated data saved to {combined_path}")
                print(f"DATA SUCCESS: Final dataset has {len(combined_df)} time periods and {combined_df.shape[1]} variables")
                
                return combined_df
            except Exception as e:
                print(f"DATA ERROR: Failed to merge aggregated datasets: {e}")
                # Return whichever dataset has more data
                return gkg_agg if len(gkg_agg) > len(events_agg) else events_agg
        else:
            # Return whichever dataset has data
            return gkg_agg if len(gkg_agg) > 0 else events_agg
    
    # If no aggregation specified, return raw data
    return {'gkg': gkg_df, 'events': events_df}

# Execute the data processing with daily aggregation
if __name__ == "__main__":
    # Options: 'D' (daily), 'H' (hourly), 'min' (original 15-min)
    aggregation_level = 'D'  
    
    # Create output directory
    output_dir = 'processed_data'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Starting GDELT data processing for Delhi with {aggregation_level} aggregation...")
    delhi_gdelt_data = process_gdelt_data(time_aggregation=aggregation_level, output_dir=output_dir)
    
    # Import power demand and weather data - Fix the filename
    power_file_path = 'weather_energy.csv'  # Updated to match your sample data
    
    power_weather_data = import_power_weather_data(
        power_file_path,
        start_date=start_date,
        end_date=end_date
    )
    
    # Optional: Try without date filters if no data found
    if len(power_weather_data) == 0:
        print("\nTrying again without date filters...")
        power_weather_data = import_power_weather_data(power_file_path)
        
        if len(power_weather_data) > 0:
            print(f"Found {len(power_weather_data)} records with time range: {power_weather_data.index.min()} to {power_weather_data.index.max()}")
            # Use only a portion of the data if needed
            start_date_actual = power_weather_data.index.min().strftime('%Y-%m-%d')
            end_date_actual = (power_weather_data.index.min() + pd.Timedelta(days=35)).strftime('%Y-%m-%d')
            print(f"Using data range: {start_date_actual} to {end_date_actual}")
            
            power_weather_data = power_weather_data[(power_weather_data.index >= start_date_actual) & 
                                                 (power_weather_data.index <= end_date_actual)]
    
    # Combine GDELT with power demand data
    combined_data = combine_gdelt_with_power_data(
        delhi_gdelt_data,
        power_weather_data,
        time_aggregation=aggregation_level,
        output_dir=output_dir
    )
    
    print(f"Processing complete. Final dataset has {len(combined_data)} records and {combined_data.shape[1]} columns.")
    print(f"All files have been saved to the '{output_dir}' directory.")
