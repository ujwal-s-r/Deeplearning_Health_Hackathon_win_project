"""
Emotion Detection Game - Integration Module

This module provides easy integration with other Python projects.
It allows importing game data for use in emotion prediction models.
"""

import os
import json
import pandas as pd
from datetime import datetime

# Default data directory
DATA_DIR = 'game_data'

def get_game_data(limit=None, sort_by_date=True, summary_only=False):
    """
    Get all collected game data as a list of dictionaries.
    
    Args:
        limit (int, optional): Limit the number of results returned. Default is None.
        sort_by_date (bool, optional): Sort results by date. Default is True.
        summary_only (bool, optional): Return only summary data. Default is False.
        
    Returns:
        list: A list of game data dictionaries
    """
    if not os.path.exists(DATA_DIR):
        return []
    
    # Get all data files
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    
    # Sort files by date (most recent first)
    if sort_by_date:
        all_files.sort(reverse=True)
    
    # Apply limit if specified
    if limit:
        all_files = all_files[:limit]
    
    # Collect data from all files
    all_data = []
    for filename in all_files:
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            data = json.load(f)
            
            # If summary only, include only essential data
            if summary_only:
                summary_data = {
                    "timestamp": data.get("timestamp", ""),
                    "score": data.get("score", 0),
                    "features": data.get("extracted_features", {}),
                    "indicators": data.get("emotional_indicators", [])
                }
                all_data.append(summary_data)
            else:
                all_data.append(data)
    
    return all_data

def get_game_data_as_dataframe(features_only=False):
    """
    Get all game data as a pandas DataFrame for easy analysis.
    
    Args:
        features_only (bool, optional): Include only extracted features. Default is False.
        
    Returns:
        pandas.DataFrame: A DataFrame containing game data
    """
    game_data = get_game_data(summary_only=True)
    
    if not game_data:
        return pd.DataFrame()
    
    if features_only:
        # Create a DataFrame with only the extracted features
        features_list = []
        timestamps = []
        
        for data in game_data:
            if "features" in data and data["features"]:
                features = data["features"]
                features_list.append(features)
                timestamps.append(data.get("timestamp", ""))
        
        df = pd.DataFrame(features_list)
        df['timestamp'] = timestamps
        return df
    else:
        # Convert to DataFrame
        df = pd.DataFrame(game_data)
        
        # Extract nested features into columns
        if "features" in df.columns:
            # Get all feature keys
            all_feature_keys = set()
            for features in df["features"]:
                if isinstance(features, dict):
                    all_feature_keys.update(features.keys())
            
            # Create a column for each feature
            for key in all_feature_keys:
                df[f"feature_{key}"] = df["features"].apply(
                    lambda x: x.get(key, None) if isinstance(x, dict) else None
                )
        
        return df

def get_emotional_indicators(limit=None):
    """
    Get emotional indicators from game data.
    
    Args:
        limit (int, optional): Limit the number of results returned. Default is None.
        
    Returns:
        list: A list of dictionaries containing emotional indicators
    """
    game_data = get_game_data(limit=limit, summary_only=True)
    
    indicators_list = []
    for data in game_data:
        if "indicators" in data and data["indicators"]:
            timestamp = data.get("timestamp", "")
            score = data.get("score", 0)
            
            for indicator in data["indicators"]:
                indicator_data = {
                    "timestamp": timestamp,
                    "score": score,
                    "emotion": indicator.get("emotion", ""),
                    "confidence": indicator.get("confidence", 0),
                    "indicators": indicator.get("indicators", [])
                }
                indicators_list.append(indicator_data)
    
    return indicators_list

def get_latest_game_data():
    """
    Get the most recent game data.
    
    Returns:
        dict: The most recent game data or None if no data exists
    """
    data_list = get_game_data(limit=1)
    return data_list[0] if data_list else None

def get_latest_features():
    """
    Get features from the most recent game.
    
    Returns:
        dict: The extracted features from the most recent game or None if no data exists
    """
    game_data = get_latest_game_data()
    if game_data and "extracted_features" in game_data:
        return game_data["extracted_features"]
    return None

def get_feature_names():
    """
    Get all available feature names.
    
    Returns:
        list: A list of feature names
    """
    game_data = get_latest_game_data()
    if game_data and "extracted_features" in game_data:
        return list(game_data["extracted_features"].keys())
    return []

if __name__ == "__main__":
    # Example usage:
    print("Available features:", get_feature_names())
    
    # Get latest game data
    latest_data = get_latest_game_data()
    if latest_data:
        print(f"Latest game played at: {latest_data.get('timestamp', 'Unknown')}")
        print(f"Score: {latest_data.get('score', 0)}")
        
        # Show emotional indicators
        if "emotional_indicators" in latest_data:
            print("\nEmotional indicators:")
            for indicator in latest_data["emotional_indicators"]:
                print(f"- {indicator['emotion']} ({int(indicator['confidence']*100)}%)")
    else:
        print("No game data available.") 