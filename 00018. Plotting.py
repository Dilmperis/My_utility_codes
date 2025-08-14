import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_event_frame(events_gen_1, bboxes=None, res_x=304, res_y=240, duration=0, pause_time=0.5):
    """
    Plots a frame of events in a 50ms chunk. 
    
    Arguments:
        - events -- a numpy array with fields 't', 'x', 'y', and 'p' representing time, x-coordinate, y-coordinate, and polarity respectively.
        - bboxes:
            An array((t, x, y, w, h, class_id, class_confidence, track_id), 
                    dtype=[('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')])
        - frame_size -- a tuple of the size of the frame (width, height).   
    """
    # Create a blank image with correct dimensions (height, width, 3)
    frame = np.zeros((res_y, res_x, 3), dtype=np.uint8)
    
    x = events_gen_1['x']
    y = events_gen_1['y']
    p = events_gen_1['p']

    # Build polarity map: (y, x) → set of polarities
    polarity_map = {}

    for i in range(len(p)):
        key = (y[i], x[i])  # Use (y, x) format for numpy indexing
        polarity_map.setdefault(key, set()).add(int(p[i]))  # convert p[i] to plain int

    # Color the pixels: Red for polarity 0, Green for polarity 1 & Yellow for both polarities in the same pixel
    '''
    SOS detail: In numpy arrays follow the convention:
        - frame[rows, columns] == frame[y, x]
        
        ========================================================|
        |Coordinate System  |    Order	 |   Example            |
        |-------------------------------------------------------|        
        |Cartesian	        |   (x, y)	 |   (240, 100)         |
        |NumPy Array Index  |	[y, x]	 |   frame[100, 240]    |
        ========================================================|
    '''
    for (y_pix, x_pix), polarities in polarity_map.items():
        if polarities == {0}:
            frame[y_pix, x_pix] = [255, 0, 0]       # Red
        elif polarities == {1}:
            frame[y_pix, x_pix] = [0, 255, 0]       # Green
        else:
            frame[y_pix, x_pix] = [255, 255, 0]     # Yellow

    
    # BBOXES:
    '''
    The format of the bboxes in plt.patches.Rectangle is:
        Rectangle((x_min, y_min), width, height)
    '''
    if len(bboxes)>0: print(f'  -bboxes: {bboxes}\n') 


    # # Display the frame
    # plt.imshow(frame)
    
    # plt.title(f'Event Frame \nVideo duration: {duration/1000000:.0f}/60s') if duration else plt.title('Event Frame') 
    # plt.pause(0.0005)  # Show for 50 ms
    # plt.clf()
    ## To display the frame without blocking the execution, use plt.pause
    ## plt.show() would block the execution until the window is closed!!!!!
    
    
    
    ## Create subplots for displaying multiply videos:
    if not plt.fignum_exists(1):
        fig, ax = plt.subplots(num=1, figsize=(10, 8))
    else:
        fig = plt.figure(1)
        ax = fig.axes[0]
    
    '''
    ==================================================================
    What is fig in fig, ax = plt.subplots()?
        - fig = Figure object
        - It represents the entire canvas or "big container" that holds all subplots, titles, legends, images, etc.
        
        Figure-level Settings:
            1)  overall size (figsize)
            2)  Add a main title (fig.suptitle("Title"))
            3)  Save the whole figure (fig.savefig("output.png"))

    Think of it like this:
        - Figure (fig) → The page
        - Axes (ax1, ax2) → The individual drawings/plots on that page
    ==================================================================
    '''
    ax.clear()  # Clear the axes to avoid overlapping plots

    ax.imshow(frame)
    ax.set_title(f'Event Frame\nVideo duration: {duration/1e6:.0f}/60s' if duration else 'Event Frame')

    for bbox in bboxes:
        x_min = bbox['x']
        y_min = bbox['y']
        width = bbox['w']
        height = bbox['h']
        label = bbox['class_id']
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

        label = "Car" if label == 0 else "Pedestrian" if label == 1 else f"Unknown ({label})"
        ax.text(x_min, y_min - 5, f'{label}', 
                color='blue', fontsize=16, 
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=1, 
                          edgecolor='none', boxstyle='round,pad=0.2'))

    plt.pause(pause_time)
