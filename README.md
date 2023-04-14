# Steps to install local machine

Install anaconda /miniconda. If you are using Mac - then the setup is a bit tricky. Below are the steps for windows.

1) Install anaconda -  https://docs.anaconda.com/anaconda/install/windows/

Make sure your installation is good by running the command in your command line - conda --version - this should print you the version

2) Get final project starter code into you local

3) Open terminal and navigate to your starter code folder

4) run command - conda env create -f environment.yml 

5) This should setup all required libraries in your local.

6) Download Pycharm and copy .idea folder from this repo into you root folder.

7) Open pycharm and open the folder - it should open up the entire workspace.

8) Configure Pycharm to point to you newly python interpreter.

File --> Settings. --> search python interpreter --> on right panel it should show Add new interpreter --> select conda environment and look up for your newly created conda environment.


That should be it.The .idea folder should expose all the required run configurations.


# Optional - Pycharm navigation and Debug
Below are useful resources to understand pycharm env , if its new to you. This is very helfpul and will help through out your MSCS program

https://www.youtube.com/watch?v=sRGpvbhOhQs&ab_channel=TechWithTim




# Steps to get Local machine running 


## Steps to get local machine to work

1) Clone project into your local/google collab
2) Use anaconda/collab env to setup supertux environment - similar to HW5
3) run train.py - From my local- D:\MSCS\DL\final\state_agent\train.py



## Unit tests
Run - PlayerTests.py which will directly run the player so that we can focus on control opearations.


## Default location of video output 

Below location in utils.py can be changed to other intended location instead of /tmp/

def show_video(frames, fps=30):     
imageio.mimwrite('/tmp/test.mp4', frames, fps=fps, bitrate=1000000)     
display(Video('/tmp/test.mp4', width=800, height=600, embed=True))
