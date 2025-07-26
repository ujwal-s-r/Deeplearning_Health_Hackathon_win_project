gemini.json->  n8n file run on n8n, <br> copy the webhook to final_result.html<br>
cd hack_1<br>
install requirements.txt<br>
python run.py<br>
https://miro.com/app/board/uXjVIDBlQsw=/?share_link_id=675100232031-> for detailed analysis<br>
app-> has the main app code<br>
collaborated repo link->https://github.com/AdvayaHackathon/151.runtime_error<br>
--ujwaljeevan123@gmail.com
--21rajkiran@gmail.com




# Mental Health Assessment Platform

This project is a comprehensive mental health assessment tool that uses a multi-modal approach to gather insights. It combines a standardized questionnaire (PHQ-8), a gamified cognitive assessment, and video-based emotional analysis to create a holistic overview. The final results are synthesized by a generative AI to produce a detailed report.

---

## Features

-   **Multi-Phase Assessment:**
    1.  **Phase 1: PHQ-8 Questionnaire:** A standard self-reporting tool for screening depression.
    2.  **Phase 2: Interactive Game:** A gamified task to measure cognitive and behavioral patterns like reaction time, accuracy, emotional bias, and performance under distraction.
    3.  **Phase 3: Video Analysis:** Uses a webcam to analyze facial expressions, blink rate, and gaze patterns while the user watches a stimulus video.
-   **AI-Powered Reporting:** Integrates with Google's Gemini AI via an n8n webhook to generate a comprehensive, professional-style report based on the collected data from all phases.
-   **Flask Backend:** Built with Python and Flask to handle application logic, data processing, and serving content.
-   **Dynamic Frontend:** Uses vanilla JavaScript to manage the interactive game, webcam recording, and asynchronous communication with the backend.

---

## Project Structure

```
Deeplearning_Health_Hackathon_win_project/
├── app/                  # Main Flask application
│   ├── __init__.py       # Application factory
│   ├── main/             # Main blueprint for application routes
│   │   ├── __init__.py
│   │   └── routes.py     # Defines all application URLs and logic
│   ├── models/           # Contains the depression prediction model
│   ├── static/           # Static assets (CSS, JS, images, videos)
│   │   ├── css/
│   │   └── js/
│   │       └── game.js   # Core logic for the interactive game
│   ├── templates/        # HTML templates for all pages
│   └── video_processor/  # Modules for video analysis (emotion, blink, gaze)
├── run.py                # Entry point to start the application
├── requirements.txt      # Python dependencies
├── gemini.json           # n8n workflow for AI report generation
└── readme.md             # Project documentation
```

---

## Execution Flow

The user journey is designed as a sequential, three-part assessment:

1.  **Start:** The user lands on the homepage ([`app/templates/index.html`](app/templates/index.html)) and begins the assessment.
2.  **PHQ-8 Questionnaire:** The user answers 8 questions. The score is calculated and stored in the user's session ([`app/main/routes.py#phq8_questionnaire`](app/main/routes.py)).
3.  **Interactive Game:** The user plays a game where they control a dot to collect rewards and avoid obstacles. The game logic in [`app/static/js/game.js`](app/static/js/game.js) tracks various metrics. At the end of the game, the data is sent to the `/save_game_data` endpoint, processed, and saved to the session.
4.  **Video Analysis:** The user is recorded via their webcam while watching a video. The recorded video is sent to the `/save_webcam_recording` endpoint. The backend analyzes the video for emotional cues, blink rate, and gaze direction, and saves the results to the session.
5.  **Comprehensive Results:** The [`/final_result`](app/main/routes.py) page displays a summary of data collected from all three phases.
6.  **AI Report Generation:**
    *   On the results page, the user can click a button to generate an AI report.
    *   This action sends all the collected data to a pre-configured n8n webhook (defined in [`gemini.json`](gemini.json)).
    *   The n8n workflow forwards the data to the Google Gemini API for analysis.
    *   The AI-generated report is returned to the user's browser and displayed on the [`/ai_report`](app/main/routes.py) page.

---

## How to Run

### Prerequisites

-   Python 3.x
-   An `n8n` instance (cloud or self-hosted) to run the AI workflow.

### 1. Setup the Backend

First, clone the repository and install the required Python packages.

```sh
# Navigate to the project directory
cd Deeplearning_Health_Hackathon_win_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup the n8n Workflow

The AI report generation relies on an n8n workflow.

1.  Open your n8n instance.
2.  Import the [`gemini.json`](gemini.json) file to create the workflow.
3.  The workflow uses a **Webhook** node. Copy the **Test URL** from this node.
4.  You will also need to configure the **Google Gemini Chat Model** node with your own Google AI API credentials.

### 3. Configure the Webhook URL

Paste the copied n8n webhook URL into the `fetch` request in the final results template.

````javascript
// filepath: app/templates/final_result.html
// ...existing code...
            // Send data to webhook
            fetch('YOUR_N8N_WEBHOOK_URL_HERE', {
                method: 'POST',
// ...existing code...