# RoadLaneDetection
Road Lane Detection using OpenCV and additional algorithms.  
One of my past projects involved autonomous vehicle movement with a minimal number of sensors involved. The entire project included a complex multi-level algorithm that involved computer vision algorithms, several neural networks (like Yolo, SCNN, etc.), and the use of additional sensors in addition to the video camera (for example, several TOFs).
The presented part of the project will use only computer vision algorithms (OpenCV) to demonstrate its capabilities (as well as limitations). However, for the case of clear standard markings, these algorithms are sufficient to find the traffic lane.  
## Example result
![til](/RoadLineDetection/data/outputs/example_video_out.gif)
Below I will describe a few highlights of how you can achieved this.
# Algorithm blocks
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
# ImageReformer
## Make Binary Image
### Standard image convertions
Grayscaling and clache normalization:  
![til](/RoadLineDetection/data/debug_images/dbg_2.jpg)  
(original image: dbg_2.jpg)  
![til](/RoadLineDetection/data/outputs/Image_reformer/1_gray_normalize.jpg)    
Separate S channel from HSL color model (we can modify sun light effect case later via S channel):  
![til](/RoadLineDetection/data/outputs/Image_reformer/1_s_channel.jpg)  
(original image: dbg_3.jpg)  
### Threshold image convertions
#### 1. Filtered S channel
**A**: Sobel threshed S-channel  
**B**: Gray Mask (check source code for explanation)  
**A - B**: Filtered S channel. We delete garbage edges from the road zone  
![til](/RoadLineDetection/data/outputs/Image_reformer/2_s_sobel-grayreger.jpg)  
(original image: dbg_3.jpg)    
#### 2. Filtered Gray SobelX  
![til](/RoadLineDetection/data/debug_images/dbg_1.jpg)  
(original image: dbg_1.jpg)  
**A**: Sobel threshed gray normalized image  
**B**: White Mask (check source code for explanation)  
**A - B**: Filtered SobelX. We delete garbage nonzero pixels
![til](/RoadLineDetection/data/outputs/Image_reformer/2_gray_sobel-no-white.jpg)
#### 3. Color filter (yellow & white)
Highlight yellow and white road lines
![til](/RoadLineDetection/data/outputs/Image_reformer/2_yellow-white_filter.jpg) 
### Additional image convertions
#### 1. Shadow filter
![til](/RoadLineDetection/data/debug_images/dbg_2.jpg)  
(original image: dbg_2.jpg)    
**A**: Filtered Gray SobelX (check THreshold Image Conversions #2)  
**B**: Shadow filter (sobel for dark regions, check source code for explanation)  
**A | B**: Final Sobel Filter.  
![til](/RoadLineDetection/data/outputs/Image_reformer/3_shadow_filter.jpg) 
As you can see, we catched lines in sun and dark regions together.  
#### 2. Light filter
This is the part of color filter.  
So it is the equivalent of shadow filter for light regions.  
TO DO: add a good example for it.  
__________________________________
So finally I combined *Sobel Filter*, *Color Filter* and *S-filter* to the general output.

