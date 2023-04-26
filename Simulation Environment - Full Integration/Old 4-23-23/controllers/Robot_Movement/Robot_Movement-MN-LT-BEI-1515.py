"""Robot_Movement controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import Dyanamics_and_Controls as dc
import VisOdom_integration as VisOdom
import numpy as np
import Sensor_Fusion as sf
import copy
import xlwt
from xlwt import Workbook

if __name__ == "__main__":

    # create the Robot instance.
    robot = Robot()
    kf = sf.KalmanFilter()
    
    # get the time step of the current world.
    timestep = 32*3 #timestep in ms
    max_speed = 6.28 #max speed in rads/s. (not needed for automation code)
    
    #Device initiation  
    #Create Lidar instance
    lidar = robot.getDevice('RPlidar A2')
    lidar.enable(timestep)
    
    #Create camera instances
    left_camera = robot.getDevice('MultiSense S21 left camera')
    left_camera.enable(timestep)
    left_ps = robot.getDevice('PS_1')
    left_ps.enable(timestep)

    right_camera = robot.getDevice('MultiSense S21 right camera')
    right_camera.enable(timestep)
    right_ps = robot.getDevice('PS_2')
    right_ps.enable(timestep)

    ps_values = [0, 0]

    imu = robot.getDevice('MultiSense S21 inertial unit')
    imu.enable(timestep)
        
    #Create LPS instance
    gps = robot.getDevice('gps')
    gps.enable(timestep)

    #Create UltraSonic instance
    #ultrasonic = robot.getDevice('UltraSonic')
    #ultrasonic.enable(timestep)
    #pointlight = robot.getDevice('PointLight')
    
    #Create gyro instance
    gyro = robot.getDevice('gyro')
    gyro.enable(timestep)
    
    #Created Motor instances
    left_motor = robot.getDevice('Motor1')
    left_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    
    right_motor = robot.getDevice('Motor2')    
    right_motor.setPosition(float('inf'))
    right_motor.setVelocity(0.0)
    
    #Create robot controller instance
    control = dc.Robot_Controller(timestep)
    kinematics = dc.Robot_Kinematics(timestep)
  
    #Initiative Variables
    current_state = np.array([0.0,0.0,0.0,0.0,0.0]) # State matrices need to change to a 5x1 matrix
    error_state = np.array([0.0,0.0,0.0,0.0,0.0]) # State matrices need to change to a 5x1 matrix
    encoder_state = np.array([0.0,0.0,0.0,0.0,0.0]) # State matrices need to change to a 5x1 matrix
    desired_velocities = np.array([1.43,0])  # Velocity matrices need to change to a 2x1 matrix. Provided by the local planner (WP2 - Local Planner)  

    # initialize variables for linear velocity calculation
    prev_orientation = np.zeros(3)
    velocity = np.zeros(3)
    position = np.zeros(3)
    prev_accel = np.zeros(3)
    # Initialize Kalman filter state
    prev_estimate = 0.0
    prev_error = 1.0  
        
    i = 0
      
    vo = VisOdom.VisualOdom(left_camera)
    
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')

    # ----- SENSOR FUSION -----    
    # Define the initial state and covariance matrices
    state = [0, 0] # [left_speed, right_speed]
    P = [[1, 0], [0, 1]] # covariance matrix
    
    # Define the process noise covariance matrix
    Q = [[0.01, 0], [0, 0.01]]
    
    # Define the measurement noise covariance matrix
    R = [[0.1, 0], [0, 0.1]]     
    
    prev_left_speed = 0
    prev_right_speed = 0
    alpha = 0.75 # Smoothing parameter    
    # ----- SENSOR FUSION -----

    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        #Initialization of state based on current values
        if i == 0:
            orientation = imu.getRollPitchYaw()
            position = gps.getValues()
            speed = gps.getSpeed()
            ang_vel = gyro.getValues()
            current_state[0] = position[0]
            current_state[1] = position[1]
            current_state[2] = orientation[2]
            current_state[3] = 0
            current_state[4] = 0
            ps_values[0]=left_ps.getValue()
            ps_values[1]=right_ps.getValue()
            initial_encoder = ps_values
            #print(initial_encoder)
            ideal = copy.deepcopy(current_state)
            for j in range(5):
                sheet1.write(i, j+1, ideal[j]) #Ideal State
                sheet1.write(i, j+6, ideal[j]) #Encoder State
                sheet1.write(i, j+11, ideal[j]) #Visual Odometry State
                sheet1.write(i, j+16, ideal[j]) #Sensor Fusion State
                
                wb.save('xlwt example.xls')

        # Pulling Lidar point cloud information    
        range_image = lidar.getRangeImage()
        
        #Pulling camera information
        left_camera.getImage()
        right_camera.getImage()
        orientation = imu.getRollPitchYaw()   
        
        #if len(vo.images) < 2:
            #vo.loadImage(left_camera)
        #else:
            #q1,q2 = vo.matchMaker(left_camera)
            #currentPose, vel, eul = vo.solvePose(q1,q2,currentPose)          
            #print(currentPose, vel, eul)
            
        #Pulling GPS information
        position = gps.getValues()
        speed = gps.getSpeed()
        
        #Pulling gyro information
        ang_vel = gyro.getValues()
        
        #Pulling encoder information
        ps_values[0]=left_ps.getValue()
        ps_values[1]=right_ps.getValue()
        final_encoder = ps_values

        #Checks - Print outputs
        #print(final_encoder)
        #print(orientation)
        #print(position)
        #print(speed)
        #print(ang_vel)
        #print(len(range_image))
        ideal[0] = position[0]
        ideal[1] = position[1]
        ideal[2] = orientation[2]
        ideal[3] = speed
        ideal[4] = ang_vel[2]
            
        #Dynamics and Controls function
        left_encoder_chg = final_encoder[0]-initial_encoder[0]
        right_encoder_chg = final_encoder[1]-initial_encoder[1]
        encoder_state = kinematics.encoder_state_estimate(current_state,left_encoder_chg,right_encoder_chg)
        

        # ----- SENSOR FUSION -----
        # Define the input variables
        visual_odometry = encoder_state #CHANGE LATER TO DEMARGIOS STUFF [1, 2, 3, 4, 5]
        initial_encoder_state = encoder_state#[6, 7, 8, 9, 10]
        
        # Call the sensor fusion function
        adjusted_encoder_state = sf.sensor_fusion(timestep, visual_odometry, initial_encoder_state)
                
        # Print the adjusted encoder state
        #print(adjusted_encoder_state)

        ## SENSOR FUSION GOES HERE
        # Incorporate 'initial encoder_state', 'visual odo' to give best wheel velocities

        # ----- SENSOR FUSION -----
                       
        current_state, error_state, wheel_vel_ideal = control.LQR(adjusted_encoder_state,error_state,desired_velocities)
        #print(wheel_vel_ideal)
        for j in range(5):
            sheet1.write(i+1, j+1, ideal[j]) #Ideal State
            sheet1.write(i+1, j+6, encoder_state[j]) #Encoder State
            sheet1.write(i+1, j+11, ideal[j]) #Visual Odometry State
            sheet1.write(i+1, j+16, ideal[j]) #Sensor Fusion State
            
            wb.save('xlwt example.xls')
        #print(encoder_state)
        #print(current_state)
        #print(wheel_vel_ideal)
        #Establishing Wheel speeds
        left_speed = wheel_vel_ideal[0]
        right_speed = wheel_vel_ideal[1]
        
        #left_speed = 0.5 * max_speed
        #right_speed = 0.3 * max_speed
        
        # ----- KALMAN FILTER -----
        # get sensor data and pass it to the update method
        print([i, left_speed, right_speed])
        kf_left_speed, kf_right_speed = kf.update(left_speed, right_speed)
        #print([i, kf_left_speed, kf_right_speed])
        # ----- KALMAN FILTER -----

        # ----- SENSOR FUSION -----
        #print([i, left_speed, right_speed])
        #print("before smoothing", [left_speed, right_speed])
        smoothed_left_speed, smoothed_right_speed = sf.smooth_velocity(left_speed, right_speed, prev_left_speed, prev_right_speed, alpha)
        #print("after smoothing", [smoothed_left_speed, smoothed_right_speed])
        #print([i, smoothed_left_speed, smoothed_right_speed])
        
        # Max caps
        if(smoothed_left_speed > 25):
            smoothed_left_speed = 25
        if(smoothed_right_speed > 25):
            smoothed_right_speed = 25       
        
        # Seting smoothed values only
        left_motor.setVelocity(smoothed_left_speed)
        right_motor.setVelocity(smoothed_right_speed)
        
        # Update the previous velocities
        prev_left_speed = smoothed_left_speed
        prev_right_speed = smoothed_right_speed
        # ----- SENSOR FUSION -----                

        #Update previous encoder information
        initial_encoder = copy.deepcopy(final_encoder)
        i += 1
    # Enter here exit cleanup code.
