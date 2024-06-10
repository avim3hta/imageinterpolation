import cv2

def process_video(input_video_path, output_video_path, zoom_factor):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply 8x zoom using Lanczos4 interpolation
        height, width = frame.shape[:2]
        zoomed_frame = cv2.resize(frame, (width * zoom_factor, height * zoom_factor), interpolation=cv2.INTER_LANCZOS4)
        
        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (zoomed_frame.shape[1], zoomed_frame.shape[0]))
        
        out.write(zoomed_frame)
    
    cap.release()
    out.release()
    print(f"Successfully processed the video and saved to {output_video_path}")

# Parameters
input_video_path = 'img/output_1280x960.avi'  # Replace with your video path
output_video_path = 'zoomed_video_lanczos.avi'  # Replace with your desired output path
zoom_factor = 8

# Process the video
process_video(input_video_path, output_video_path, zoom_factor)
