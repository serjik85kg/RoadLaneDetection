# RoadLaneDetection
Road Lane Detection using OpenCV and additional algorithms.  
One of my past projects on my first work involved autonomous vehicle movement with a minimal number of sensors involved. The entire project included a complex multi-level algorithm that involved computer vision algorithms, several neural networks (like Yolo, SCNN, etc.), and the use of additional sensors in addition to the video camera (for example, several TOFs).
The presented part of the project will use only computer vision algorithms (OpenCV) to demonstrate its capabilities (as well as limitations). However, for the case of clear standard markings, these algorithms are sufficient to find the traffic lane.  
## Requirements
 - C++17 standard
 - Opencv 3
 - Eigen
## Example result
![til](/RoadLineDetection/data/outputs/example_video_out.gif)
Below I will describe a few highlights of how I achieveded this.
# Algorithm blocks
  - ImageReformer
      - Make binary image
        - standard image convertions (gray, normalization, RGB->HSL, etc.)
        - threshold image convertions (sobel, color)
        - additional image convertions (processing dark regions(shadow filter) and light regions (light filter))
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
# Part 1. ImageReformer
This part will consist of special binary image conversions using the Opencv library methods.
## Make Binary Image
### Standard image convertions
Grayscaling and clache normalization:  
![til](/RoadLineDetection/data/debug_images/dbg_2.jpg)  
(original image: *dbg_2.jpg*)  
![til](/RoadLineDetection/data/outputs/Image_reformer/1_gray_normalize.jpg)    
Separate S channel from HSL color model (we can modify sun light effect case later via S channel):  
![til](/RoadLineDetection/data/outputs/Image_reformer/1_s_channel.jpg)  
(original image: *dbg_3.jpg*)  
### Threshold image convertions
#### 1. Filtered S channel
**A**: Sobel threshed S-channel  
**B**: Gray Mask (check source code for explanation)  
**A - B**: Filtered S channel. We delete garbage edges from the road zone  
![til](/RoadLineDetection/data/outputs/Image_reformer/2_s_sobel-grayreger.jpg)  
(original image: *dbg_3.jpg*)    
#### 2. Filtered Gray SobelX  
![til](/RoadLineDetection/data/debug_images/dbg_1.jpg)  
(original image: *dbg_1.jpg*)  
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
(original image: *dbg_2.jpg*)    
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
![til](/RoadLineDetection/data/debug_images/dbg_4.jpg)  
(original image: *dbg_4.jpg*)  
![til](/RoadLineDetection/data/outputs/Image_reformer/combined_binary.jpg)
![til](/RoadLineDetection/data/outputs/Image_reformer/Colored_binary.jpg) 
Sobel Filter - Green  
Color Filter - Blue  
S-Filter - Red  
**So now we have good binary output image, and we can see road lanes in interested area. We must try to catch lane points separately. Next step: Lane Handle!**  

# Part 2. Lane Handle
In this part, we will find the traffic lane from the converted binary image.
### Image transforms
One of the main additional transformations is **image warping**.  
We set the initial and final shape of the quadrilateral area of the image (the road zone) and find the forward and reverse transformation matrices.  
![til](/RoadLineDetection/data/outputs/Lane_handle/warped_example.jpg)  
For more information, see the code in the **laneHandle::transforms** namespace.
### Detect lane pixels
To detect lane pixels we used sliding boxes algorithm (check **laneHandle::findLanePixels(...)**).
We set the starting positions of the boxes, calculate the center of mass of the points inside them, and move the box so that the center of mass of the points is in the center of the box.   
Repeat this for all boxes for each line.  
The points "captured" inside these boxes will belong to our lines.
Then, for each array of line points, we calculate an approximate quadratic function.  
Previously, opencv had a special built-in function for this, but it was removed in opencv 3, so I added my own implementation (for more information, see **eigenOperations::polyfit(...)**).  
![til](/RoadLineDetection/data/outputs/Lane_handle/warped_fit_polynomial.jpg)
Left line points are red. Right line points are blue.   
Each was searched using search boxes that are marked by green (in this case, 9 on each side).  
After all, using the two arrays of points found on the left and right lines, the approximation of each of them to a quadratic function was calculated (yellow lines).  
### Find linear perspective
In this block we find perspective lines and quadrilateral area from this lines for road lane detecting.  
It is a part of adaptive rectification algorithm. To recalculate the source points of perspective rectangle we approximated lines on original image using least squares algorithm (see **laneHandle::findLinearLines()**). To get lane points we used perspective transform from warped to original view, as points on warped image were found by sliding box or search from prior algorithms (check it later).  
After lines approximation is found, we look for their cross, otherwise it is considered as bad detection. Then we calculate source points of perspective rectangle (**laneHandle::findPespectiveRect()**). This calculation is parametized by width of rectangle's. This is useful to control how much distance ahead will perspective transform fetch.
See the the output images.
![til](/RoadLineDetection/data/outputs/Lane_handle/find_perspective.jpg)
Lines apporximation and their cross is shown with blue color.  
Perspective rectangle is shown with red color.
#### Search Around Poly
This is an auxiliary function that will be used in parallel with **laneHandle::findLanePixels(...)**, depending on the number of frames lost (more on this in the next section).  
![til](/RoadLineDetection/data/outputs/Lane_handle/warped_around_poly.jpg)
### Get traffic lane
Fillpoly between detected lines and unwrap to original image.
![til](/RoadLineDetection/data/outputs/Lane_handle/draw_lane.jpg)
Additional: calculate radius of lines and slide of central position (metrics were taken randomly, just for example).  

# Part 3. Processor (Main class)
This class combines and binds the algorithms described above, handles cases of false and good ones, calculates average values for several frames,and draws the results.
Its constructor has the following parameters:
 - doAverage (bool). If set to True, averaged polynomial lane approximation will be drawn (if use metrics, averaged values of curvature radius will be shown)  
 - doDebug (bool). I used this for debugging. Now deprecated.
 - defCorners (std::vector<cv::Point2f>). Default locations of source points for perspective transform. Use for initial and reset.
 - defWidth (int). Default width of upper edge of perspective rectangle's top edge. Use for initial and reset.  
 
And main inference method **Processor::Inference(const cv::Mat& src, bool isDebug = false)**:  
The following pipeline is establihed there:
 - Make Binary
 - Warp image (do perspective transorm) using current perspective matrix
 - Detect lane using either 'sliding window search' or 'search around poly' depending on value of **Processor::m_lostFrames** which tracks number of not detected frames
 - Test if detection was good (**Processor::isGoodLane()**). If lane is good, process it with **Processor::processGoodLane()**, otherwise process it with **Processor::processBadLane()**. These functions set values of Processor::m_lLane and Processor::m_rLane objects, which have custom type *Line* (check Processor.h).
 - Calculate and show average values of parameters we are interested in. In this case i show only position in release mode and some others in debug mode. For this calculation historical values are utilized (see members of class Line: Line::m_recentXFitted, Line::m_recentFit). If there is no history (Processor::m_historyLen = 0) the corresponding message is shown on the frame.
**Processor::processGoodLane()** controls the viewport of perspective transform. If we approach a sharp turn (**Processor::isSharpTurn(...)**), it decreases value of **Processor::m_perspRectTop**; if we see flat lane (**Processor::IsFlatLane(...)**), it increases it. If perspective transformation matrix was changed, we recalculate all historical values, which depend on it, according to the new viewport (see **Processor::UpdatePerspective()**).

# Conclusion
A small note.
As I wrote earlier, this project is only part of a large and complex project. The algorithm doesn't take into account difficult urban situations, asphalt with puddles, and can't handle glare from the sun and headlights of oncoming cars. And it is impossible to foresee all these cases with ordinary computer vision, using only one camera. It's brutal do get all information only from color of pixels. That's why machine learning is so popular nowadays. However, this algorithm works well for motorways and other roads where there are two clear marking lines for each lane.
Ultimately, my goal here was to demonstrate the capabilities of computer vision, the OpenCV library, and how to apply functions from this library.  
By the way, since the project had to be written "from memory", I didn't bother much with optimization. I will return to this question as soon as I have the strength and time. This will not affect the understanding and logic of the algorithm in any way.
As soon as I have time, I will expand this project by adding additional features that can find the road in an urban environment, even in some cases with non-standard markings.
### Bonus brutal video output
![til](/RoadLineDetection/data/outputs/example_video_hard.gif)  
Pretty good :)
