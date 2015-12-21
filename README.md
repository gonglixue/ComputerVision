# ComputerVision
use opencv to finish some simple assignments and realize some fundamental computer vision algorithms.
There are several projects in this responsity. Simple introduction of each is following:
## Project 1:Personal Photo Album
This is my first opencv project in the course Computer Vision.
This program can generate a personal photo video using the photos and video you input. In another word, the program join your photos and video together in one video. You should put your photos and video(only one) in a folder. And then you should rename the photos in such format " 0.jpg, 1.jpg, 2.jpg, 3.jpg, 4.jpg". The only video should be renamed as "video.avi". Execute cmd under this folder, and input "hw1_V2.exe 'the path of your resource folder'". And finally you'll see a video named "output.avi" in the same folder.

## Project 2: Harris Corner Detection
It is a command line program to detect corners in a picture. Some explaination of arguments:
argv[1]: the picture file name you want to detect corners.
argv[2]: represents "k" in formula.
argv[3]: the aperture size of sobel.
This program realize the harris corner detection algorithm. The main part of this algorithm is self-completed without using the function relevant to harris corner in opencv. But for convenience, the operation about matrix calculation is executed by function in opencv.
