import cv2
import numpy as np

# Video characteristics
frame_height, frame_width = 480, 640
fps = 30 
video_duration = 10 #secs
total_frames = fps*video_duration

# Video implementation
cv2.namedWindow("Random Video Stream", cv2.WINDOW_NORMAL) # It makes the window resizable

for frame in range(total_frames):
    random_frame = np.random.randint(0, 256, (frame_height, frame_width, 3), dtype=np.uint8)
    cv2.imshow("Random Video Stream", random_frame)
    cv2.waitKey(int(1000/fps))

    if (frame+1)%100==0:
        print(f'{frame+1}/{total_frames} frames')

        
cv2.destroyAllWindows()
