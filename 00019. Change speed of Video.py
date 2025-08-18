from moviepy.editor import VideoFileClip, vfx

def change_video_speed(input_path, output_path, speed_factor=0.5):
    """ Change Video speed by the given speed factor.
    
    - input_path: str, path to the input mp4 file
    - output_path: str, path to save the slowed video
    - speed_factor: float, e.g. 0.5 means half speed (slower),
                  2.0 means double speed (faster)
    """
    clip = VideoFileClip(input_path)  # Load the video
    slowed_clip = clip.fx(vfx.speedx, factor=speed_factor) # Change Speed
    slowed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac") # Write to output

    # Close resources
    clip.close()
    slowed_clip.close()

if __name__ == "__main__":
    input_video = "Downloads\\Media2.mp4"
    output_video = "Downloads\\Media2_changed.mp4"
    change_video_speed(input_video, output_video, speed_factor=0.10)
