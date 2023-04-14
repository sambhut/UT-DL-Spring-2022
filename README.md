1) Clone project into your local/google collab
2) Use anaconda/collab env to setup supertux environment - similar to HW5
3) run train.py - From my local- D:\MSCS\DL\final\state_agent\train.py


Unit tests
Run - PlayerTests.py which will directly run the player so that we can focus on control opearations.


Default location of video output - /tmp/ can be changed to other location

def show_video(frames, fps=30):     
imageio.mimwrite('/tmp/test.mp4', frames, fps=fps, bitrate=1000000)     
display(Video('/tmp/test.mp4', width=800, height=600, embed=True))
