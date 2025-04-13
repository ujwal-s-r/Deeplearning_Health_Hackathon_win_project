from flask import Flask, render_template, jsonify, request, send_file
import json
import os
import math
import statistics
import csv
from datetime import datetime

app = Flask(__name__)

# Create data directory if it doesn't exist
DATA_DIR = 'game_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/save_game_data', methods=['POST'])
def save_game_data():
    # Get data from request
    data = request.json
    
    # Add timestamp
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract features for emotion prediction
    features = extract_features(data)
    data['extracted_features'] = features
    
    # Analyze emotional indicators
    emotional_indicators = analyze_emotional_indicators(features)
    data['emotional_indicators'] = emotional_indicators
    
    # Save data to file
    filename = f"{DATA_DIR}/game_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved game data to {filename}")
    print(f"Extracted features: {features}")
    print(f"Emotional indicators: {emotional_indicators}")
    
    return jsonify({
        "status": "success",
        "features": features,
        "emotional_indicators": emotional_indicators,
        "message": "Game data saved successfully"
    })

@app.route('/api/export_data', methods=['GET'])
def export_data():
    """
    Export all collected game data as JSON for integration with other systems.
    Query parameters:
    - format: 'json' or 'csv' (default: 'json')
    - summary_only: 'true' or 'false' (default: 'false')
    """
    export_format = request.args.get('format', 'json')
    summary_only = request.args.get('summary_only', 'false').lower() == 'true'
    
    # Get all data files
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    all_files.sort(reverse=True)  # Most recent first
    
    if not all_files:
        return jsonify({"status": "error", "message": "No data found"}), 404
    
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
    
    # Export based on format
    if export_format == 'csv':
        # Create a temporary CSV file
        csv_file = os.path.join(DATA_DIR, "export_data.csv")
        export_to_csv(all_data, csv_file, summary_only)
        return send_file(csv_file, as_attachment=True, download_name="game_data_export.csv")
    else:
        # Return JSON
        return jsonify({
            "status": "success",
            "count": len(all_data),
            "data": all_data
        })

def export_to_csv(data_list, output_file, summary_only=False):
    """Export data to CSV format"""
    if not data_list:
        return
    
    # Determine fields to export
    if summary_only:
        # First level fields
        fields = ["timestamp", "score"]
        
        # Add feature fields if they exist
        if "features" in data_list[0] and data_list[0]["features"]:
            feature_fields = sorted(data_list[0]["features"].keys())
            
            # Handle emotion indicator fields
            emotion_fields = []
            if "indicators" in data_list[0] and data_list[0]["indicators"]:
                emotion_fields = ["emotion_indicators"]
    else:
        # For full export, use all fields from the first item
        # This is simplified and will only include first level fields
        fields = sorted(data_list[0].keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if summary_only:
            # Write header with feature columns
            header = fields + ["feature_" + f for f in feature_fields] + emotion_fields
            writer.writerow(header)
            
            # Write each row
            for item in data_list:
                row = [item.get(f, "") for f in fields]
                
                # Add feature values
                for f in feature_fields:
                    row.append(item.get("features", {}).get(f, ""))
                
                # Add emotion indicators as a formatted string
                if "indicators" in item and item["indicators"]:
                    emotions = "; ".join([f"{i['emotion']} ({int(i['confidence']*100)}%)" 
                                         for i in item["indicators"]])
                    row.append(emotions)
                else:
                    row.append("")
                    
                writer.writerow(row)
        else:
            # Simple flat export of first level fields
            writer.writerow(fields)
            for item in data_list:
                writer.writerow([json.dumps(item.get(f, "")) for f in fields])

def extract_features(data):
    """
    Extract relevant features from game data for emotion prediction.
    These features can be used as input to your emotion detection model.
    """
    # Basic performance metrics (Phase 1 & 2)
    basic_metrics = {
        "score": data.get("score", 0),
        "stars_collected": data.get("starsCollected", 0),
        "obstacles_avoided": data.get("obstacleAvoidances", 0),
        "missed_rewards": data.get("missedRewards", 0),
        "avg_reward_reaction_time": data.get("averageRewardReactionTime", 0),
        "avg_obstacle_reaction_time": data.get("averageObstacleReactionTime", 0),
        "positive_emoji_interactions": data.get("positiveEmojiInteractions", 0),
        "negative_emoji_interactions": data.get("negativeEmojiInteractions", 0),
        "neutral_emoji_interactions": data.get("neutralEmojiInteractions", 0),
        "distraction_response_delta": data.get("distractionResponseDelta", 0),
    }
    
    # New Phase 3 metrics
    movement_metrics = {
        "movement_direction_changes": data.get("movementDirectionChanges", 0),
        "hesitations": data.get("hesitations", 0),
        "movement_variability": data.get("movementVariability", 0),
        "avg_hesitation_duration": data.get("avgHesitationDuration", 0),
        "hesitation_frequency": data.get("hesitationFrequency", 0),
        "direction_change_frequency": data.get("directionChangeFrequency", 0),
    }
    
    emotional_response_metrics = {
        "avg_response_to_positive": data.get("avgResponseToPositive", 0),
        "avg_response_to_negative": data.get("avgResponseToNegative", 0),
        "emotional_response_ratio": data.get("emotionalResponseRatio", 0),
    }
    
    # Calculate derived features from raw data
    derived_features = {
        "collection_efficiency": calculate_collection_efficiency(data),
        "distraction_recovery": calculate_distraction_recovery(data),
        "emotional_bias": calculate_emotional_bias(data),
    }
    
    # Advanced Phase 3 derived features
    advanced_features = {}
    
    # Analyze reaction patterns to different emotions
    reaction_patterns = analyze_reaction_patterns(data)
    
    # Analyze movement patterns after distractions
    movement_analysis = analyze_movement_patterns(data)
    
    # Analyze hesitation patterns
    hesitation_analysis = analyze_hesitation_patterns(data)
    
    # Add new requested features (Phase 4)
    additional_features = {
        # 1. Reaction Time Variability
        "reaction_time_variability": calculate_reaction_time_variability(data),
        
        # 2. Performance Degradation
        "performance_degradation": calculate_performance_degradation(data),
        
        # 3. Emotional Stimuli Avoidance Rate
        "emotional_stimuli_avoidance_rate": calculate_emotional_avoidance_rate(data),
        
        # 4. Emotional Type Preferences
        "emoji_preference_profile": calculate_emoji_preference_profile(data),
    }
    
    # Combine all features
    features = {**basic_metrics, **movement_metrics, **emotional_response_metrics, 
               **derived_features, **advanced_features, **reaction_patterns, 
               **movement_analysis, **hesitation_analysis, **additional_features}
    
    return features

def calculate_collection_efficiency(data):
    """Calculate how efficiently the player collected rewards"""
    stars_collected = data.get("starsCollected", 0)
    missed_rewards = data.get("missedRewards", 0)
    total_rewards = stars_collected + missed_rewards
    
    if total_rewards == 0:
        return 0
    
    return stars_collected / total_rewards

def calculate_distraction_recovery(data):
    """Calculate recovery rate after distractions"""
    distraction_events = data.get("distractionEvents", [])
    
    if not distraction_events:
        return 0
    
    # Average speed change (negative means slowing down after distraction)
    speed_changes = [event.get("speedDelta", 0) for event in distraction_events]
    if not speed_changes:
        return 0
    
    return sum(speed_changes) / len(speed_changes)

def calculate_emotional_bias(data):
    """
    Calculate emotional bias score
    Positive value indicates preference for positive stimuli
    Negative value indicates preference for negative stimuli
    """
    positive = data.get("positiveEmojiInteractions", 0)
    negative = data.get("negativeEmojiInteractions", 0)
    total = positive + negative
    
    if total == 0:
        return 0
    
    # Range from -1 (all negative) to 1 (all positive)
    return (positive - negative) / total

def analyze_reaction_patterns(data):
    """Analyze patterns in reactions to different emotional stimuli"""
    emotional_responses = data.get("emotionalStimulusResponses", [])
    
    if not emotional_responses:
        return {
            "reaction_consistency": 0,
            "emotional_preference_strength": 0,
            "reaction_time_variability": 0
        }
    
    # Group reactions by emotion type
    reaction_by_type = {}
    for response in emotional_responses:
        emotion_type = response.get("type", "")
        if emotion_type not in reaction_by_type:
            reaction_by_type[emotion_type] = []
        reaction_by_type[emotion_type].append(response)
    
    # Calculate reaction time consistency (lower standard deviation = more consistent)
    all_reaction_times = [response.get("reactionTime", 0) for response in emotional_responses]
    reaction_time_std = statistics.stdev(all_reaction_times) if len(all_reaction_times) > 1 else 0
    max_std = 2000  # Arbitrary high value to normalize against
    reaction_consistency = 1 - min(reaction_time_std / max_std, 1)
    
    # Calculate preference strength (how strongly they prefer one emotion over others)
    counts = {emotion: len(responses) for emotion, responses in reaction_by_type.items()}
    total = sum(counts.values())
    proportions = {emotion: count / total for emotion, count in counts.items()}
    # Gini coefficient as a measure of inequality (higher = stronger preference)
    emotional_preference_strength = calculate_gini_coefficient(list(proportions.values()))
    
    # Calculate reaction time variability between different emotion types
    avg_reaction_times = {}
    for emotion, responses in reaction_by_type.items():
        times = [r.get("reactionTime", 0) for r in responses]
        avg_reaction_times[emotion] = sum(times) / len(times) if times else 0
    
    # If we have both positive and negative emotion reaction times
    if 'happy' in avg_reaction_times and ('sad' in avg_reaction_times or 'angry' in avg_reaction_times):
        happy_time = avg_reaction_times['happy']
        sad_time = avg_reaction_times.get('sad', 0)
        angry_time = avg_reaction_times.get('angry', 0)
        
        # Use the available negative emotion, prioritize sad if both are available
        negative_time = sad_time if sad_time > 0 else angry_time
        
        # Calculate normalized difference (positive values mean faster reactions to positive)
        if happy_time > 0 and negative_time > 0:
            reaction_time_variability = (negative_time - happy_time) / ((negative_time + happy_time) / 2)
        else:
            reaction_time_variability = 0
    else:
        reaction_time_variability = 0
    
    return {
        "reaction_consistency": reaction_consistency,
        "emotional_preference_strength": emotional_preference_strength,
        "reaction_time_variability": reaction_time_variability
    }

def analyze_movement_patterns(data):
    """Analyze player movement patterns, especially after distractions"""
    distraction_events = data.get("distractionEvents", [])
    idle_periods = data.get("idlePeriods", [])
    
    # Calculate percentage of idle periods that occurred after distractions
    post_distraction_idle = 0
    if idle_periods:
        for period in idle_periods:
            # Check if the idle period started within 2 seconds of a distraction
            if period.get("timeSinceLastDistraction", float('inf')) < 2000:
                post_distraction_idle += 1
        
        post_distraction_idle_ratio = post_distraction_idle / len(idle_periods) if idle_periods else 0
    else:
        post_distraction_idle_ratio = 0
    
    # Get average speed change after distractions
    avg_speed_delta = 0
    if distraction_events:
        speed_deltas = [event.get("speedDelta", 0) for event in distraction_events]
        avg_speed_delta = sum(speed_deltas) / len(speed_deltas)
    
    # Calculate consistency of distraction response
    consistency = 0
    if distraction_events and len(distraction_events) > 1:
        speed_deltas = [event.get("speedDelta", 0) for event in distraction_events]
        # Standard deviation of speed changes normalized to the mean
        mean_delta = abs(sum(speed_deltas) / len(speed_deltas)) if sum(speed_deltas) != 0 else 1
        std_delta = statistics.stdev(speed_deltas) if len(speed_deltas) > 1 else 0
        # Lower STD/mean ratio means more consistent responses
        consistency = 1 - min(std_delta / (mean_delta * 3), 1) if mean_delta > 0 else 0
    
    return {
        "post_distraction_idle_ratio": post_distraction_idle_ratio,
        "avg_post_distraction_speed_change": avg_speed_delta,
        "distraction_response_consistency": consistency
    }

def analyze_hesitation_patterns(data):
    """Analyze player hesitation patterns"""
    idle_periods = data.get("idlePeriods", [])
    
    if not idle_periods:
        return {
            "avg_hesitation_duration": 0,
            "hesitation_frequency": 0,
            "hesitation_pattern_consistency": 0
        }
    
    # Average duration of hesitations
    durations = [period.get("duration", 0) for period in idle_periods]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    # Frequency of hesitations per second
    game_time = data.get("gameTime", 60)  # Default to 60 seconds if not provided
    frequency = len(idle_periods) / game_time
    
    # Consistency of hesitation durations (lower standard deviation = more consistent)
    hesitation_std = statistics.stdev(durations) if len(durations) > 1 else 0
    # Normalize STD against the mean
    consistency = 1 - min(hesitation_std / (avg_duration * 2), 1) if avg_duration > 0 else 0
    
    return {
        "avg_hesitation_duration": avg_duration,
        "hesitation_frequency": frequency,
        "hesitation_pattern_consistency": consistency
    }

def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient as a measure of inequality
    Used to measure preference strength (1 = strong preference, 0 = equal preference)
    """
    if not values or sum(values) == 0:
        return 0
    
    # Sort values
    sorted_values = sorted(values)
    height, area = 0, 0
    
    for value in sorted_values:
        height += value
        area += height - value / 2
    
    fair_area = height * len(values) / 2
    return (fair_area - area) / fair_area if fair_area > 0 else 0

def calculate_reaction_time_variability(data):
    """
    Calculate overall reaction time variability (standard deviation)
    across all interactions to measure consistency
    """
    # Get all reaction times from the data
    reaction_times = []
    
    # Include reaction times from rewards
    reaction_time_data = data.get("reactionTimeDetail", [])
    for item in reaction_time_data:
        if isinstance(item, dict) and "time" in item:
            reaction_times.append(item["time"])
    
    # Calculate standard deviation if we have enough data points
    if len(reaction_times) > 1:
        try:
            return statistics.stdev(reaction_times)
        except statistics.StatisticsError:
            return 0  # Default if calculation fails
    return 0  # Default if not enough data

def calculate_performance_degradation(data):
    """
    Calculate performance change over time (beginning vs. end of game)
    Negative values indicate degradation, positive values indicate improvement
    """
    # Get reaction time details and sort by time
    reaction_details = data.get("reactionTimeDetail", [])
    if not reaction_details or len(reaction_details) < 4:  # Need enough data points
        return 0
    
    try:
        # Sort by time (assuming each entry has a timestamp or sequence)
        # This assumes the entries in reactionTimeDetail are in chronological order
        first_quarter = reaction_details[:len(reaction_details)//4]
        last_quarter = reaction_details[-len(reaction_details)//4:]
        
        # Calculate average reaction times for first and last quarter
        first_quarter_times = [r.get("time", 0) for r in first_quarter if isinstance(r, dict)]
        last_quarter_times = [r.get("time", 0) for r in last_quarter if isinstance(r, dict)]
        
        if not first_quarter_times or not last_quarter_times:
            return 0
            
        first_avg = sum(first_quarter_times) / len(first_quarter_times)
        last_avg = sum(last_quarter_times) / len(last_quarter_times)
        
        # Calculate normalized difference (negative means degradation/slower)
        # We invert the value so positive means improvement (faster reactions)
        if first_avg > 0:
            return (first_avg - last_avg) / first_avg
        return 0
    except (TypeError, ZeroDivisionError):
        return 0

def calculate_emotional_avoidance_rate(data):
    """
    Calculate tendency to avoid negative emotional stimuli
    Higher values indicate greater avoidance of negative emojis
    """
    # Get emotional stimulus responses
    emotional_responses = data.get("emotionalStimulusResponses", [])
    
    # Count interactions with different emoji types
    happy_interactions = 0
    sad_interactions = 0
    angry_interactions = 0
    neutral_interactions = 0
    
    # Count missed emojis of each type (if available)
    happy_missed = 0
    sad_missed = 0
    angry_missed = 0
    neutral_missed = 0
    
    # Count interactions
    for response in emotional_responses:
        if not isinstance(response, dict):
            continue
            
        emoji_type = response.get("type", "")
        if emoji_type == "happy":
            happy_interactions += 1
        elif emoji_type == "sad":
            sad_interactions += 1
        elif emoji_type == "angry":
            angry_interactions += 1
        elif emoji_type == "neutral":
            neutral_interactions += 1
    
    # If we don't have missed emoji data, we can approximate using the
    # relative frequencies compared to happy emojis (assuming equal generation rates)
    total_interactions = happy_interactions + sad_interactions + angry_interactions + neutral_interactions
    
    if total_interactions == 0:
        return 0
        
    # Calculate percentage of negative emoji interactions out of all interactions
    negative_interactions = sad_interactions + angry_interactions
    positive_interactions = happy_interactions
    
    # If no positive or negative interactions, can't calculate ratio
    if positive_interactions == 0 or negative_interactions == 0:
        return 0
        
    # Calculate ratio of positive to negative interactions
    # Higher values indicate more avoidance of negative emojis
    ratio = positive_interactions / negative_interactions
    
    # Normalize to 0-1 range (using a reasonable maximum ratio of 5)
    normalized_ratio = min(ratio / 5, 1)
    
    return normalized_ratio

def calculate_emoji_preference_profile(data):
    """
    Calculate a detailed breakdown of emoji interaction patterns
    Returns a dictionary with normalized preference scores for each emoji type
    """
    # Get emotional stimulus responses
    emotional_responses = data.get("emotionalStimulusResponses", [])
    
    # Count interactions and reaction times for each emoji type
    emoji_data = {
        "happy": {"count": 0, "reaction_times": []},
        "sad": {"count": 0, "reaction_times": []},
        "angry": {"count": 0, "reaction_times": []},
        "neutral": {"count": 0, "reaction_times": []}
    }
    
    # Collect data for each emoji type
    for response in emotional_responses:
        if not isinstance(response, dict):
            continue
            
        emoji_type = response.get("type", "")
        reaction_time = response.get("reactionTime", 0)
        
        if emoji_type in emoji_data:
            emoji_data[emoji_type]["count"] += 1
            if reaction_time > 0:
                emoji_data[emoji_type]["reaction_times"].append(reaction_time)
    
    # Calculate average reaction times
    for emoji_type in emoji_data:
        times = emoji_data[emoji_type]["reaction_times"]
        emoji_data[emoji_type]["avg_reaction_time"] = sum(times) / len(times) if times else 0
        
    # Calculate interaction preferences (percentage of total interactions)
    total_interactions = sum(emoji_data[emoji_type]["count"] for emoji_type in emoji_data)
    
    if total_interactions == 0:
        # Return default values if no interactions
        return {
            "happy_preference": 0,
            "sad_preference": 0,
            "angry_preference": 0,
            "neutral_preference": 0,
            "positive_preference": 0,
            "negative_preference": 0
        }
        
    # Calculate preference scores (normalized to 0-1)
    preference_profile = {
        "happy_preference": emoji_data["happy"]["count"] / total_interactions,
        "sad_preference": emoji_data["sad"]["count"] / total_interactions,
        "angry_preference": emoji_data["angry"]["count"] / total_interactions,
        "neutral_preference": emoji_data["neutral"]["count"] / total_interactions
    }
    
    # Add aggregated preferences
    preference_profile["positive_preference"] = preference_profile["happy_preference"]
    preference_profile["negative_preference"] = (
        preference_profile["sad_preference"] + preference_profile["angry_preference"]
    )
    
    return preference_profile

def analyze_emotional_indicators(features):
    """
    Analyze features to identify potential emotional indicators.
    Returns a list of possible emotional states with confidence levels.
    """
    indicators = []
    
    # Check for anxiety indicators
    anxiety_score = calculate_anxiety_score(features)
    if anxiety_score > 0.3:  # Threshold for reporting
        indicators.append({
            "emotion": "Anxiety",
            "confidence": anxiety_score,
            "indicators": get_anxiety_indicators(features, anxiety_score)
        })
    
    # Check for depression indicators
    depression_score = calculate_depression_score(features)
    if depression_score > 0.3:  # Threshold for reporting
        indicators.append({
            "emotion": "Depression",
            "confidence": depression_score,
            "indicators": get_depression_indicators(features, depression_score)
        })
    
    # Check for general emotional arousal
    arousal_score = calculate_arousal_score(features)
    if arousal_score > 0.7:  # High arousal
        indicators.append({
            "emotion": "High Emotional Arousal",
            "confidence": arousal_score,
            "indicators": ["High movement variability", "Quick reaction times", "Strong emotional responses"]
        })
    elif arousal_score < 0.3:  # Low arousal
        indicators.append({
            "emotion": "Low Emotional Arousal",
            "confidence": 1 - arousal_score,
            "indicators": ["Low movement variability", "Slow reaction times", "Weak emotional responses"]
        })
    
    # Check for emotional valence (positive vs negative)
    valence_score = calculate_valence_score(features)
    if valence_score > 0.7:  # Positive emotional state
        indicators.append({
            "emotion": "Positive Emotional State",
            "confidence": valence_score,
            "indicators": ["Preference for positive stimuli", "Higher scores", "Quick reactions to positive items"]
        })
    elif valence_score < 0.3:  # Negative emotional state
        indicators.append({
            "emotion": "Negative Emotional State",
            "confidence": 1 - valence_score,
            "indicators": ["Preference for negative stimuli", "Lower scores", "Quicker reactions to negative items"]
        })
    
    return indicators

def calculate_anxiety_score(features):
    """
    Calculate an anxiety score based on game features
    Higher values suggest higher anxiety
    """
    indicators = []
    
    # High reaction to negative stimuli (lower time = faster reaction = higher score)
    negative_bias = 0
    if features["reaction_time_variability"] < 0:
        negative_bias = min(abs(features["reaction_time_variability"]), 1)
    
    # High movement variability
    movement_var = min(features["movement_variability"] / 100, 1)
    
    # Many direction changes
    direction_changes = min(features["direction_change_frequency"] * 2, 1)
    
    # Increased speed after distractions (hypervigilance)
    distraction_effect = 0
    if features["avg_post_distraction_speed_change"] > 0:
        distraction_effect = min(features["avg_post_distraction_speed_change"] * 5, 1)
    
    # High inconsistency in hesitation patterns
    hesitation_inconsistency = 1 - features["hesitation_pattern_consistency"]
    
    # Combine indicators (weighted)
    anxiety_score = (
        negative_bias * 0.25 +
        movement_var * 0.2 +
        direction_changes * 0.2 +
        distraction_effect * 0.2 +
        hesitation_inconsistency * 0.15
    )
    
    return min(max(anxiety_score, 0), 1)  # Ensure between 0 and 1

def get_anxiety_indicators(features, score):
    """Get human-readable anxiety indicators based on the features"""
    indicators = []
    
    if features["reaction_time_variability"] < -0.2:
        indicators.append("Faster reactions to negative stimuli")
        
    if features["movement_variability"] > 50:
        indicators.append("High movement variability (frequent adjustments)")
        
    if features["direction_change_frequency"] > 0.5:
        indicators.append("Frequent direction changes (potential restlessness)")
        
    if features["avg_post_distraction_speed_change"] > 0.05:
        indicators.append("Increased speed after distractions (hypervigilance)")
        
    if features["hesitation_pattern_consistency"] < 0.3:
        indicators.append("Inconsistent hesitation patterns")
        
    if len(indicators) == 0:
        indicators.append("General anxiety indicators present")
        
    return indicators

def calculate_depression_score(features):
    """
    Calculate a depression score based on game features
    Higher values suggest higher depression indicators
    """
    # Low overall activity (movement variability)
    low_activity = 1 - min(features["movement_variability"] / 50, 1)
    
    # Slow reaction times
    slow_reactions = 0
    avg_time = features["avg_reward_reaction_time"]
    if avg_time > 800:  # Assuming 800ms is a threshold for slow reactions
        slow_reactions = min((avg_time - 800) / 1000, 1)
    
    # More missed rewards (low motivation)
    missed_ratio = 1 - features["collection_efficiency"]
    
    # Longer hesitations
    long_hesitations = min(features["avg_hesitation_duration"] / 2000, 1)  # Normalize to 2 seconds
    
    # Negative emotional bias
    negative_bias = (1 - features["emotional_bias"]) / 2  # Convert from [-1,1] to [0,1]
    
    # Slowing down after distractions (rumination)
    post_distraction_slowing = 0
    if features["avg_post_distraction_speed_change"] < 0:
        post_distraction_slowing = min(abs(features["avg_post_distraction_speed_change"]) * 10, 1)
    
    # Combine indicators (weighted)
    depression_score = (
        low_activity * 0.2 +
        slow_reactions * 0.2 +
        missed_ratio * 0.15 +
        long_hesitations * 0.15 +
        negative_bias * 0.15 +
        post_distraction_slowing * 0.15
    )
    
    return min(max(depression_score, 0), 1)  # Ensure between 0 and 1

def get_depression_indicators(features, score):
    """Get human-readable depression indicators based on the features"""
    indicators = []
    
    if features["movement_variability"] < 30:
        indicators.append("Low movement variability (reduced activity)")
        
    if features["avg_reward_reaction_time"] > 800:
        indicators.append("Slower reaction times (psychomotor retardation)")
        
    if features["collection_efficiency"] < 0.5:
        indicators.append("Low reward collection (potential reduced motivation)")
        
    if features["avg_hesitation_duration"] > 1000:
        indicators.append("Extended hesitations (decision-making difficulty)")
        
    if features["emotional_bias"] < -0.2:
        indicators.append("Negative emotional bias")
        
    if features["avg_post_distraction_speed_change"] < -0.05:
        indicators.append("Slowdown after distractions (potential rumination)")
        
    if len(indicators) == 0:
        indicators.append("General depression indicators present")
        
    return indicators

def calculate_arousal_score(features):
    """
    Calculate emotional arousal score (how emotionally activated/engaged)
    Higher values indicate higher arousal
    """
    # Fast reaction times indicate high arousal
    reaction_speed = 0
    if features["avg_reward_reaction_time"] > 0:
        # Normalize: lower times = higher scores (up to 1)
        reaction_speed = min(2000 / features["avg_reward_reaction_time"], 1)
    
    # High movement variability indicates high arousal
    movement_activity = min(features["movement_variability"] / 100, 1)
    
    # Frequent direction changes indicate high arousal
    direction_activity = min(features["direction_change_frequency"] * 2, 1)
    
    # Strong preference for any emotional stimuli indicates high arousal
    emotional_engagement = features["emotional_preference_strength"]
    
    # Combine indicators
    arousal_score = (
        reaction_speed * 0.3 +
        movement_activity * 0.3 +
        direction_activity * 0.2 +
        emotional_engagement * 0.2
    )
    
    return min(max(arousal_score, 0), 1)  # Ensure between 0 and 1

def calculate_valence_score(features):
    """
    Calculate emotional valence score (positive vs negative emotional state)
    Higher values indicate more positive emotional state
    """
    # Emotional bias toward positive stimuli
    emotional_bias = (features["emotional_bias"] + 1) / 2  # Convert from [-1,1] to [0,1]
    
    # Collection efficiency (higher = more positive)
    efficiency = features["collection_efficiency"]
    
    # Faster reactions to positive vs negative stimuli
    valence_reaction = (features["reaction_time_variability"] + 1) / 2  # Convert to [0,1]
    
    # Score (normalized to 0-1)
    score_valence = 0
    max_possible_score = 60  # Rough estimate of max possible score
    if features["score"] > 0:
        score_valence = min(features["score"] / max_possible_score, 1)
    
    # Combine indicators
    valence_score = (
        emotional_bias * 0.35 +
        efficiency * 0.25 +
        valence_reaction * 0.2 +
        score_valence * 0.2
    )
    
    return min(max(valence_score, 0), 1)  # Ensure between 0 and 1

# Add a route to display available data
@app.route('/data_viewer')
def data_viewer():
    """Simple data viewer for game data"""
    # Get all data files
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    all_files.sort(reverse=True)  # Most recent first
    
    data_summaries = []
    for filename in all_files:
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            data = json.load(f)
            
            # Create a summary
            summary = {
                "filename": filename,
                "timestamp": data.get("timestamp", "Unknown"),
                "score": data.get("score", 0),
                "indicators": [i["emotion"] for i in data.get("emotional_indicators", [])]
            }
            data_summaries.append(summary)
    
    return render_template('data_viewer.html', data_summaries=data_summaries)

if __name__ == '__main__':
    app.run(debug=True) 