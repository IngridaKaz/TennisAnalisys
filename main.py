from utils import (read_video,
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    #Read video
    input_video_path = "input_videos/berankis03.mp4"
    video_frames = read_video(input_video_path)

    #Detecting Players and ball
    player_tracker = PlayerTracker(model_path='models/yolov8x')
    ball_tracker = BallTracker(model_path='models/yolo5_best.pt')
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    #Court Line Detector model
    court_model_path = "models/keypoints_model_resnet50.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)


    #### Draw outputs

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    #draw court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()