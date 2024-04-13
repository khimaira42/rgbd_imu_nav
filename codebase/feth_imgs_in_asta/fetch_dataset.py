import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Get IMU data
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        #rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(color_image, depth_image)
        depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        #depth_trunc = pipeline.get_active_profile().get_device().first_depth_sensor().get_option(rs.option.depth_units)
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        #print("depth_intrin:"+str(depth_intrin))
    

        d_time = depth_frame.get_timestamp()/1000
        print("Depth frame timestamp:"+str(d_time))
        # get the current time in seconds since the epoch
        seconds = time.time()
        print("Seconds since epoch =", seconds)
        # convert the time in seconds since the epoch to a readable format
        local_time = time.ctime(seconds)
        print("Local time:", local_time)
        
        # Convert depth image to 8-bit grayscale
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        # Convert grayscale image to 3-channel RGB image
        depth_image_rgb = cv2.cvtColor(depth_image_8bit, cv2.COLOR_GRAY2RGB)
        concatenated_image = np.hstack((color_image, depth_image_rgb))
        cv2.imshow("Color and Depth Images", concatenated_image)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:

    # Stop streaming
    pipeline.stop()
