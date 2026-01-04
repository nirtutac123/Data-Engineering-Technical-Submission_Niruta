#!/usr/bin/env python3
"""
Feature Engineering Module
--------------------------
Creates advanced features from cleaned crime data for modeling.
Includes temporal, spatial, and categorical transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os

def load_cleaned_data(city='chicago', nrows=None):
    """Load cleaned data for a specific city."""
    base_path = os.path.join('..', '..', 'data', 'clean')
    if city == 'chicago':
        # Load the most recent Chicago file
        file_path = os.path.join(base_path, 'chicago_chicago_crimes_2012_to_2017_clean.csv')
    elif city == 'la_crimes':
        file_path = os.path.join(base_path, 'la_crimes_crime-data-from-2010-to-present_clean.csv')
    else:
        raise ValueError("Unsupported city")

    df = pd.read_csv(file_path, nrows=nrows)
    return df

def temporal_features(df):
    """Extract temporal features from incident_datetime."""
    df = df.copy()
    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'], errors='coerce')

    # Basic time features
    df['hour'] = df['incident_datetime'].dt.hour
    df['day_of_week'] = df['incident_datetime'].dt.dayofweek
    df['month'] = df['incident_datetime'].dt.month
    df['year'] = df['incident_datetime'].dt.year

    # Derived features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=['winter', 'spring', 'summer', 'fall'])

    return df

def spatial_features(df):
    """Create spatial features using coordinates."""
    df = df.copy()

    # Basic coordinate features
    df['lat_rounded'] = df['latitude'].round(2)
    df['lon_rounded'] = df['longitude'].round(2)

    # Clustering for location groups (if coordinates available)
    coords = df[['latitude', 'longitude']].dropna()
    if len(coords) > 1000:
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
        df['location_cluster'] = np.nan
        df.loc[coords.index, 'location_cluster'] = kmeans.fit_predict(coords)

    return df

def categorical_features(df):
    """Encode and transform categorical features."""
    df = df.copy()

    # Frequency encoding for high-cardinality categoricals
    for col in ['crime_subtype', 'location_type', 'block_address']:
        if col in df.columns:
            freq = df[col].value_counts()
            df[f'{col}_freq'] = df[col].map(freq)

    # Label encoding for ordinal features
    ordinal_cols = ['district', 'ward', 'community_area']
    for col in ordinal_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df

def aggregate_features(df):
    """Create aggregate features by grouping."""
    df = df.copy()

    # Crime rate by district and hour
    if 'district' in df.columns and 'hour' in df.columns:
        district_hour_crimes = df.groupby(['district', 'hour']).size().reset_index(name='district_hour_crime_count')
        df = df.merge(district_hour_crimes, on=['district', 'hour'], how='left')

    # Rolling crime counts (if we had time series data)
    # For now, just add a simple count feature

    return df

def scale_features(df, numerical_cols):
    """Scale numerical features."""
    df = df.copy()
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def create_features(city='chicago', nrows=None, target_col='arrest_made'):
    """Main function to create all features."""
    print(f"Loading {city} data...")
    df = load_cleaned_data(city, nrows)

    print("Creating temporal features...")
    df = temporal_features(df)

    print("Creating spatial features...")
    df = spatial_features(df)

    print("Creating categorical features...")
    df = categorical_features(df)

    print("Creating aggregate features...")
    df = aggregate_features(df)

    # Select features for modeling
    feature_cols = [
        'crime_category', 'crime_subtype', 'location_type', 'district', 'ward',
        'community_area', 'domestic_flag', 'latitude', 'longitude', 'year',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_night',
        'lat_rounded', 'lon_rounded', 'location_cluster',
        'crime_subtype_freq', 'location_type_freq', 'district_hour_crime_count'
    ]

    # Keep only available columns
    available_features = [col for col in feature_cols if col in df.columns]
    available_features.append(target_col) if target_col in df.columns else None

    df = df[available_features].dropna()

    # Encode categoricals
    cat_cols = ['crime_category', 'crime_subtype', 'location_type', 'season']
    cat_cols = [col for col in cat_cols if col in df.columns]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Convert booleans
    bool_cols = ['domestic_flag', 'is_weekend', 'is_night']
    bool_cols = [col for col in bool_cols if col in df.columns]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    if target_col in df.columns:
        df[target_col] = df[target_col].astype(int)

    # Scale numerical
    num_cols = ['latitude', 'longitude', 'year', 'hour', 'day_of_week', 'month',
                'lat_rounded', 'lon_rounded', 'crime_subtype_freq', 'location_type_freq',
                'district_hour_crime_count']
    num_cols = [col for col in num_cols if col in df.columns]
    df, scaler = scale_features(df, num_cols)

    print(f"Feature engineering complete. Shape: {df.shape}")
    return df, available_features[:-1] if target_col in available_features else available_features, scaler

def run_feature_engineering():
    """Run feature engineering for all datasets."""
    print("Running feature engineering...")

    # Process Chicago data
    try:
        df_chicago, features_chicago, scaler_chicago = create_features('chicago', nrows=50000)
        print(f"Chicago features created: {len(features_chicago)} features")
    except Exception as e:
        print(f"Error processing Chicago data: {e}")

    # Process LA crimes data
    try:
        df_la_crimes, features_la_crimes, scaler_la_crimes = create_features('la_crimes', nrows=50000)
        print(f"LA Crimes features created: {len(features_la_crimes)} features")
    except Exception as e:
        print(f"Error processing LA crimes data: {e}")

    print("Feature engineering completed.")

if __name__ == '__main__':
    # Example usage
    df, features, scaler = create_features('chicago', nrows=10000)
    print("Features created:", features[:10])
    print(df.head())
