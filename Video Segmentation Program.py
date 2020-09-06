'''
@author: Abhishek Murali and Mukul Jain
'''
#%%
import os
# To change working directory in order to run the program 
#os.chdir('Insert the path')
os.chdir('C:\Mukul\Documents\WS201920\ippr\Project 2020\Project Final\Python Program')
#%%
# Video segmentation Program
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# Start time sec
t1=627
# End time sec
t2=667
# ffmpeg_extract_subclip("Insert the path for input video", t1, t2, targetname="Inset the path for output video")
ffmpeg_extract_subclip("/content/sample_data/tagesthemen 22-15 Uhr, 26.09.2017.mp4", t1, t2, targetname="/content/sample_data/vidsegtest3.mp4")

#%%