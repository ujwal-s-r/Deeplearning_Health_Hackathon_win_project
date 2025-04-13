# Attention and Reaction-Time Dodge Game

A game that measures attention and reaction time to assess emotional responses. Players control an avatar to avoid obstacles and collect rewards, with distractions designed to measure emotional reactions.

## Phase 1 Features
- Basic gameplay with player avatar, obstacles (red squares), and rewards (green stars)
- 60-second time limit
- Score tracking
- Mobile touch support and desktop mouse controls

## Phase 2 Features
- Emotional emoji elements (happy, sad, angry, neutral) with point values
- Background flash distractions every 15 seconds
- Enhanced data collection:
  - Reaction times to stimuli
  - Movement patterns and speed changes
  - Pre/post-distraction behavior changes
  - Emotional bias detection
- Data analytics and visualization of results
- Server-side data storage for analysis

## Phase 3 Features
- Advanced movement pattern analysis:
  - Hesitation detection and analysis
  - Direction change tracking
  - Movement variability calculation
  - Idle period measurement
- Enhanced emotional response metrics:
  - Reactions to specific emotion types
  - Emotional bias strength measurement
  - Distraction recovery patterns
- Sophisticated emotional indicators:
  - Anxiety detection with confidence levels
  - Depression indicator analysis
  - Emotional arousal assessment
  - Valence determination (positive/negative state)
- Detailed data visualization and feedback

## Phase 4 Features (Current)
- Integration-ready functionality:
  - Clean API endpoint for data export (JSON/CSV)
  - Python integration module for easy import
  - Feature extraction for external models
- Data management:
  - Data viewer interface
  - Export options with different formats and levels of detail
  - Simplified data structure for integration
- Extended emotional analysis metrics:
  - **Reaction Time Variability**: Standard deviation of reaction times for consistency measurement
  - **Performance Degradation**: Score change comparing beginning vs. end of game
  - **Emotional Stimuli Avoidance Rate**: Tendency to avoid negative emoji stimuli
  - **Emoji Preference Profile**: Detailed breakdown of emoji interaction patterns and preferences
- Optimized for external system integration

## Requirements
- Python 3.x
- Flask
- Pandas (for data integration)

## Installation

1. Clone this repository
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Running the Game

1. Start the Flask server:
```
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000/
```

3. Click the "Start Game" button to begin playing

## Game Controls
- **Desktop**: Click and drag the blue dot left/right with your mouse
- **Mobile**: Touch and drag the blue dot left/right

## Game Elements
- **Blue Circle**: Player avatar
- **Red Squares**: Obstacles to avoid (-1 point)
- **Green Stars**: Rewards to collect (+1 point)
- **Emojis**:
  - 😊 Happy: +2 points
  - 😐 Neutral: 0 points
  - 😢 Sad: -1 point
  - 😠 Angry: -2 points
- **Red Flash**: Distraction that appears every 15 seconds

## Data Collection and Analysis
The game collects and analyzes the following metrics:

### Basic Metrics
- Score and performance statistics
- Reaction times to different stimuli
- Movement patterns and speed
- Response to emotional distractions
- Emotional bias (preference for positive/negative stimuli)

### Advanced Metrics
- **Movement Analysis**:
  - Direction change frequency and patterns
  - Hesitation frequency, duration, and consistency
  - Movement variability (standard deviation of positions)
  - Post-distraction movement changes

- **Emotional Response Analysis**:
  - Reaction time comparisons between positive/negative stimuli
  - Emotional preference strength (Gini coefficient)
  - Reaction consistency across emotion types
  - Emotional response ratio

- **Additional Analysis (Phase 4)**:
  - Reaction time variability (consistency in responses)
  - Performance change over time (improvement or degradation)
  - Emotional stimuli avoidance tendencies
  - Detailed emoji-type preference breakdown

- **Psychological Indicators**:
  - Anxiety indicators with confidence levels
  - Depression indicators with confidence levels
  - Emotional arousal assessment (high/low)
  - Emotional valence measurement (positive/negative)

## Integration

### Data Export
- **Web Interface**: Visit `/data_viewer` to view and export data
- **API Endpoint**: Use `/api/export_data` with the following options:
  - `format`: 'json' or 'csv' (default: 'json')
  - `summary_only`: 'true' or 'false' (default: 'false')

### Python Integration
For Python projects, use the provided `integration.py` module:

```python
# Import the integration module
from integration import get_game_data, get_game_data_as_dataframe, get_latest_features

# Get all game data
data = get_game_data()

# Get data as a pandas DataFrame
df = get_game_data_as_dataframe(features_only=True)

# Get features from the latest game
features = get_latest_features()
```

## Output
After playing, the game provides:
1. Basic performance metrics (score, stars collected)
2. Movement pattern analysis 
3. Emotional response measurements
4. Potential emotional indicators with confidence levels
5. Detailed emoji preference breakdown

The data is stored in JSON format in the `game_data` directory.

## Coming in Future Phases
- Multi-session tracking for longitudinal analysis
- Machine learning integration for emotion prediction
- Customizable difficulty levels
- More complex distraction mechanisms 