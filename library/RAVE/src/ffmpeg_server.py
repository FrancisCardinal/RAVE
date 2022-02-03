# https://stackoverflow.com/questions/68545688/stream-images-from-python-opencv-with-ffmpeg
import cv2
#import time
import subprocess as sp
import glob
import os

img_width = 1280
img_height = 720


test_path = './test_dataset'  # Folder with synthetic sample images.

os.makedirs(test_path, exist_ok=True)  # Create folder for input images.

os.chdir(test_path)

# May use full path like: 'c:\\FFmpeg\\bin\\ffmpeg.exe'
ffmpeg_cmd = 'c:\\ffmpeg\\bin\\ffmpeg.exe'
# May use full path like: 'c:\\FFmpeg\\bin\\ffplay.exe'
ffplay_cmd = 'c:\\ffmpeg\\bin\\ffplay.exe'


# Create 10 synthetic JPEG images for testing (image0001.jpg, image0002.jpg, ..., image0010.jpg).
sp.run([ffmpeg_cmd, '-y', '-f', 'lavfi', '-i',
       f'testsrc=size={img_width}x{img_height}:rate=1:duration=10', 'image%04d.jpg'])


img_list = glob.glob("*.jpg")
img_list_len = len(img_list)
img_index = 0

fps = 5

rtsp_server = 'rtsp://localhost:9090/live.stream'

# You will need to start the server up first, before the sending client (when using TCP). See: https://trac.ffmpeg.org/wiki/StreamingGuide#Pointtopointstreaming
# Use FFplay sub-process for receiving the RTSP video.
# ffplay_process = sp.Popen([ffplay_cmd, '-rtsp_flags', 'listen', rtsp_server])


command = [ffmpeg_cmd,
           '-re',
           # Apply raw video as input - it's more efficient than encoding each frame to PNG
           '-f', 'rawvideo',
           '-s', f'{img_width}x{img_height}',
           '-pixel_format', 'bgr24',
           '-r', f'{fps}',
           '-i', '-',
           '-pix_fmt', 'yuv420p',
           '-c:v', 'libx264',
           '-bufsize', '64M',
           '-maxrate', '4M',
           '-rtsp_transport', 'tcp',
           '-f', 'rtsp',
           #'-muxdelay', '0.1',
           rtsp_server]

# Execute FFmpeg sub-process for RTSP streaming
process = sp.Popen(command, stdin=sp.PIPE)


while True:
    # Read a JPEG image to NumPy array (in BGR color format) - assume the resolution is correct.
    current_img = cv2.imread(img_list[img_index])
    img_index = (img_index+1) % img_list_len  # Cyclically repeat images

    # Write raw frame to stdin pipe.
    process.stdin.write(current_img.tobytes())

    cv2.imshow('current_img', current_img)  # Show image for testing

    # time.sleep(1/FPS)
    # We need to call cv2.waitKey after cv2.imshow
    key = cv2.waitKey(int(round(1000/fps)))

    if key == 27:  # Press Esc for exit
        break


process.stdin.close()  # Close stdin pipe
process.wait()  # Wait for FFmpeg sub-process to finish
ffplay_process.kill()  # Forcefully close FFplay sub-process
cv2.destroyAllWindows()  # Close OpenCV window
