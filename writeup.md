## Project: Search and Sample Return
---

[//]: # (Image References)

[image1]: ./output/rock1.png
[image2]: ./output/rock2.png
[image3]: ./output/rock3.png
[image4]: ./output/rock4.png
[image5]: ./output/rock5.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
+ The recorded dataset settle in `test_dataset/recorded_data` folder
+ Added `color_thresh_rock(img, rgb_thresh=(110, 110, 50)):` function to identify rock samples based on thresh conditions
+ Added `mask` and identify obstacle by using mask area minus navigable area `np.absolute(np.float32(threshed) - 1) * mask`


#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
+ Threshed the current image to identify navigable terrain, obstacles, and rock samples
+ Turned pixels points to rover coordinate
+ Turned points from rover coordinate to world coordinate
+ Added world points corresponding to navigable terrain, obstacles, rock samples to the worldmap
+ Added worldmap to the output image frame 

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
The perception_step is mostly the same process_image() above, there is some improvement to make the results better:

+ `perception_step()`

```python
# To improve Fidelity, only recognize points in near range(70)
# because points in a longer range may be incorrect due to the error during perspective transformation
for idx in range(len(y_world)):
    if np.sqrt(ypix[idx]**2 + xpix[idx]**2) < 70:
        Rover.worldmap[y_world[idx], x_world[idx], 2] += 10 

for idx in range(len(y_obs_world)):
    if np.sqrt(yobs[idx]**2 + xobs[idx]**2) < 70:
        Rover.worldmap[y_obs_world[idx], x_obs_world[idx], 0] += 10 
```

```python
# If rock appear in the image frame, find the nearest point of the Rock 
# because the shorter distance points the more accuracy
if threshed_rock.any():
    rock_dist, rock_angles = to_polar_coords(xrock, yrock)
    rock_idx = np.argmin(rock_dist)
    rock_nearest_x = x_rock_world[rock_idx]
    rock_nearest_y = y_rock_world[rock_idx]
    Rover.worldmap[rock_nearest_y, rock_nearest_x, 1] = 255
    Rover.vision_image[:, :, 1] = threshed_rock * 255
    Rover.rock_angle = rock_angles[rock_idx] 
    Rover.rock_dist = rock_dist[rock_idx] 
else:
    Rover.vision_image[:, :, 1] = 0
    Rover.rock_angle = None
    Rover.rock_dist = None 
```

+ `decision_step()`

```python
# Keep track of how many times Rover is stuck in one position with velocity nearly 0
# It will help us to detect the stuck position and assist Rover find a strategy to escape
def updateStopTimes(Rover):
    if Rover.vel < 0.05:
        Rover.stop_times = Rover.stop_times + 1
    else:
        Rover.stop_times = 0 
```

```python
# Slow down the Rover by changing brake and throttle value to move gradually to the rock position
# If the Rover is near the Rock, reduce the Rover's velocity by increase brake but
# still prevent Rover to stop immediately by keeping velocity(using throttle)
# + Distance 70: still far from rock, keep speed at around 1 m/s
# + Distance 40: near the rock but not too close, keep speed at around 0.7 m/s
# + Distance 20: near the rock, keep speed at low as possible 0.4 m/s
# + Distance 10: beside the rock, should not increase speed more
# using throttle=0.4 because in some hard positions, small throttle could not force Rover move forward  
def goToRock(Rover):
    if Rover.vel < 0.5:
        Rover.brake = 0
    elif Rover.vel < 1:
        Rover.brake = 0.3
    else:
        Rover.brake = 0.6

    if Rover.rock_dist < 10:
        Rover.throttle = 0
    elif Rover.rock_dist < 20: 
        if Rover.vel < 0.4:
            Rover.throttle = 0.4
        else:
            Rover.throttle = 0
    elif Rover.rock_dist < 40:
        if Rover.vel < 0.7:
            Rover.throttle = 0.4
        else:
            Rover.throttle = 0
    elif Rover.rock_dist < 70:
        if Rover.vel < 1:
            Rover.throttle = 0.4
        else:
            Rover.throttle = 0
    else:   
        restrictMaxVel(Rover)
    
    Rover.steer = np.clip(getAngle(Rover.rock_angle), -30, 30)
```

In order to maximize the map's area that the Rover could discover, I added the simple strategy:
+ Rover will favor to navigate left side first
+ If Rover is stuck, it will turn right a small angle
```python
# Rock have the higher precedence, Rover will pick Rock first
if Rover.rock_angle is not None:
    goToRock(Rover)
# If there is no Rock in front, then discover the map by preferring left side first
else:
    restrictMaxVel(Rover)
    
    # Find indices that have distance above MIN_DIST
    # If the points is too near the Rover, the angles have no value.
    indices = [i for i in range(len(Rover.nav_dists)) if Rover.nav_dists[i] > MIN_DIST]

    # Find possible left most angle that satisfy MIN_DIST condition
    steer_angle = getAngle(np.max(Rover.nav_angles[indices])) if len(indices) else None 

    if steer_angle is not None:
        # Steer in (left most angle - 10) because of the body of the Rover
        # If we steer with left most angle we may hit the wall
        Rover.steer = np.clip(steer_angle - 10, -15, 15)
    else:
        steerMeanAngle(Rover)
```

```python
# If in a state where want to pickup a rock send pickup command
if Rover.near_sample:
    if Rover.vel == 0:
        if not Rover.picking_up:
            Rover.send_pickup = True

    # Try to stop in order to have chance of picking up the sample
    else:
        Rover.throttle = 0
        Rover.steer = 0
        Rover.brake = 10
```


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

+ Settings

Screen Resolution: 1024x768
Graphic Quality: Good
FPS: 8

+ Approach on perception

From image frame I extracted the information about the position of navigable terrain, obstacles and rock samples using perspective transform and thresholding.
Then, I converted all points to world coordinate in order to add those values to the world map using matrix transformation
In order to increase Fidelity I only map the points near Rover because those points will have a higher possibility of correct since they will not be changed 
significantly due to perspective transform.
But by using the above approach to improve Fidelity, the Rover need to take more time, more view in order to record the map entirely.

+ Approach on decision

In order to maximize the map's area that the Rover could discover, I added the simple strategy:
-Rover will favor to navigate left side first
-If Rover is stuck, it will turn right a small angle
The disadvantage of the above approach is that sometime Rover take a long time to find a left edge, when it found left edge, it will follow left edge easily
Another disadvantage is that it could not detect small obstacles like small group of rocks in the middle. We could write another algorithm to detect rock obstacles 
in the middle and calculate the maximum distance the Rover could navigate follow one angle without having any obstacle. With this information we could be smarter of 
choosing the right angle to navigate.

In order to pick up the rock, I added some breakpoints:
-It will help Rover to navigate gradually but not too slow to the rock
-Rover have time to stop if it is moving too fast
The function(breakpoints) is still very simple so the movement of Rover is not smooth

In order to avoid Rover be stuck at one position forever, I added the detection which count how many times Rover don't move(velocity equal zero) and take action.
Rover will try to steer right a small angle until it find the way to escape.
The function is still very simple since we could do better if based on the frame image we could calculate exactly could Rover move to one direction or not.
And we also could give Rover a memory to memorize the history path, if it is stuck, it could return the the first point which is still not discovered yet. 

Below is the examples steps that I experiment with my algorithm:

![picking rock 1][image1]
![picking rock 2][image2]
![picking rock 3][image3]
![picking rock 4][image4]
![picking rock 5][image5]


