from flask import Flask, send_from_directory, send_file
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hackathon_mental_health_app'
    
    # Configure file upload settings
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
    app.config['UPLOAD_EXTENSIONS'] = ['.webm', '.mp4']
    
    # Register blueprints
    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    # Multiple direct routes for the hack.mp4 file
    video_paths = [
        '/hack.mp4',
        '/video.mp4',
        '/app/vedio/hack.mp4',
        '/app/vedio/vedioforhackathon.mp4',
        '/vedio/hack.mp4',
        '/vedio/vedioforhackathon.mp4'
    ]
    
    # Get absolute path to video file
    video_file_path = os.path.join(app.static_folder, 'vedio', 'hack.mp4')
    
    # Create routes for all paths
    for path in video_paths:
        app.add_url_rule(path, f'serve_video_{path.replace("/", "_")}', 
                        lambda: send_file(video_file_path, mimetype='video/mp4'))
    
    # Direct the root to the video_analysis page
    @app.route('/direct_video')
    def direct_video():
        return send_file('static/test_video.html')
    
    return app 