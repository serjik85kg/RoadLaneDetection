# RoadLaneDetection
Road Lane Detection using OpenCV and additional algorithms.  
One of my past projects involved autonomous vehicle movement with a minimal number of sensors involved. The entire project included a complex multi-level algorithm that involved computer vision algorithms, several neural networks (like Yolo, SCNN, etc.), and the use of additional sensors in addition to the video camera (for example, several TOFs).
The presented part of the project will use only computer vision algorithms (OpenCV) to demonstrate its capabilities (as well as limitations). However, for the case of clear standard markings, these algorithms are sufficient to find the traffic lane.  
### Example result
![til](/RoadLineDetection/data/outputs/example_video_out.gif)
Below I will describe a few highlights of how you can achieved this.
## Algorithm blocks
  - ImageReformer
      - Make binary image
        - standard image convertions (gray, normalization, RGB->HSL, etc.)
        - threshold image convertions (sobel, color)
        - additional image convertions (processing dark regions: light filter, shadow filter)
  - LaneHandle
      - Image transforms
      - Detect lane pixels
      - Find linear perspective lines
      - Get traffic lane
        - Additional: calculate metrics
  - Processor (Main class)
      - Identification good/bad detections
      - Identification history
      - Multi-frame alignment
      - Get final result  
### ImageReformer
#### Standard image convertions
Grayscaling and clache normalization:
![til](/RoadLineDetection/data/outputs/Image_reformer/1_gray_normalize.jpg)
Separate S channel from HSL color model (we can modify sun light effect case later via S channel):
![til](/RoadLineDetection/data/outputs/Image_reformer/1_s_channel.jpg)
#### Threshold image convertions
A: Sobel threshed S-channel
B: Gray Mask (check source code for explanation)
A - B: Filtered S channel. We delete garbage edges from the road zone
![til](/RoadLineDetection/data/outputs/Image_reformer/2_s_sobel-grayreger.jpg)
