import numpy as np

MAX_STOP_TIMES = 10
MIN_DIST = 35
CORNER_ANGLE = 10
STEER_CHUNK = -7

def getAngle(pi_angle):
    return pi_angle * 180/np.pi    

def updateStopTimes(Rover):
    # Keep track of how many times Rover is stuck in one position with velocity nearly 0
    if Rover.vel < 0.05:
        Rover.stop_times = Rover.stop_times + 1
    else:
        Rover.stop_times = 0 

def restrictMaxVel(Rover):
    Rover.brake = 0
    if Rover.vel < 0.5:
        Rover.throttle = 0.5
    elif Rover.vel < Rover.max_vel:
        Rover.throttle = Rover.throttle_set
    else:
        Rover.throttle = 0

def goToRock(Rover):
    # Slow down the Rover by brake and throttle to move forward to the rock position
    # If the Rover is near the Rock, reduce the Rover's velocity by increase brake but
    # still prevent Rover to stop immediately
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

def steerMeanAngle(Rover):
    Rover.steer = np.clip(np.mean(getAngle(Rover.nav_angles)), -15, 15)

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            updateStopTimes(Rover)
            # Check the extent of navigable terrain or if Rock in front
            # but not when Rover is stuck in one positions for many times
            if ((len(Rover.nav_angles) >= Rover.stop_forward or 
                Rover.rock_angle is not None) and Rover.stop_times < MAX_STOP_TIMES):

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
            # If there's a lack of navigable terrain pixels
            # or Rover is stuck in on positions for many times
            # then go to 'stop' mode
            elif (len(Rover.nav_angles) < Rover.stop_forward) or (Rover.stop_times >= MAX_STOP_TIMES):
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = STEER_CHUNK
                    Rover.mode = 'stop'
                    Rover.stop_times = MAX_STOP_TIMES * 2 

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2: 
                # Now we're stopped and we have vision data to see if there's a path forward
                # Try to steer and escape from the current position
                if len(Rover.nav_angles) < Rover.go_forward or Rover.stop_times >= MAX_STOP_TIMES:
                    if Rover.vel <= 0.05:
                        Rover.stop_times = Rover.stop_times - 1
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    Rover.steer = STEER_CHUNK # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                # or If we already try hard enough by steering, then try to move forward
                else:
                    # Set throttle back to stored value
                    # Release the brake
                    Rover.brake = 0
                    Rover.throttle = Rover.throttle_set
                    Rover.mode = 'forward'
                    Rover.stop_times = 0
                    Rover.steer = STEER_CHUNK
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
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
    
    return Rover

