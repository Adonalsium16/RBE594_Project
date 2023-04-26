"""Robot_Movement controller.""" 

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import Dyanamics_and_Controls as dc
import VisOdom_integration as VisOdom
import numpy as np
import Sensor_Fusion as sf
import copy
#import xlwt
from classdef import bot
from DWAdef import DWA
#from xlwt import Workbook

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
    desired_velocities = np.array([1.43,0.0])  # Velocity matrices need to change to a 2x1 matrix. Provided by the local planner (WP2 - Local Planner)  

    # initialize variables for linear velocity calculation
    prev_orientation = np.zeros(3)
    velocity = np.zeros(3)
    position = np.zeros(3)
    prev_accel = np.zeros(3)
    # Initialize Kalman filter state
    prev_estimate = 0.0
    prev_error = 1.0  
        
    i = 0
      
    # Initialize Visual Odometry  
    vo = VisOdom.VisualOdom(left_camera, timestep)
    
    #wb = Workbook()
    #sheet1 = wb.add_sheet('Sheet 1')

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
    
    kfs = sf.KalmanFilter_State()
    # ----- SENSOR FUSION -----
    goal = [43,53]
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        #Initialization of state based on current values
        if i == 0:
            orientation = imu.getRollPitchYaw()
            position = gps.getValues()
            speed = gps.getSpeed()
            ang_vel = gyro.getValues()
            vo.loadImage(left_camera)
            current_state[0] = position[0]
            current_state[1] = position[1]
            current_state[2] = orientation[2]
            current_state[3] = 0
            current_state[4] = 0
            ps_values[0]=left_ps.getValue()
            ps_values[1]=right_ps.getValue()
            initial_encoder = ps_values
            ideal = copy.deepcopy(current_state)
            voState = copy.deepcopy(current_state)
            adjusted_encoder_state = copy.deepcopy(current_state)

            #for j in range(5):
            #    sheet1.write(i, j+1, ideal[j]) #Ideal State
            #    sheet1.write(i, j+6, ideal[j]) #Encoder State
            #    sheet1.write(i, j+11, ideal[j]) #Visual Odometry State
            #    sheet1.write(i, j+16, ideal[j]) #Sensor Fusion State
                
            #    wb.save('xlwt example.xls')

        # Pulling Lidar point cloud information    
        #range_image = lidar.getRangeImage()
        
        #Pulling camera information
        left_camera.getImage()
        right_camera.getImage()
        orientation = imu.getRollPitchYaw()   
                  
        #Pulling GPS information
        position = gps.getValues()
        speed = gps.getSpeed()
        
        #Pulling gyro information
        ang_vel = gyro.getValues()
        
        #Pulling encoder information
        ps_values[0]=left_ps.getValue()
        ps_values[1]=right_ps.getValue()
        final_encoder = ps_values
        ideal[0] = position[0]
        ideal[1] = position[1]
        ideal[2] = orientation[2]
        ideal[3] = speed
        ideal[4] = ang_vel[2]
        
        #Dynamics and Controls function
        left_encoder_chg = final_encoder[0]-initial_encoder[0]
        right_encoder_chg = final_encoder[1]-initial_encoder[1]
        encoder_state = kinematics.encoder_state_estimate(current_state,left_encoder_chg,right_encoder_chg)

        # Visual Odom Calculations
        q1,q2 = vo.matchMaker(left_camera)
        voState = vo.solvePose(q1,q2,adjusted_encoder_state) 
        
        # ************************************************************    
        # ----- SENSOR FUSION -----
        # Define the input variables - Encoder and Visual Odom thought the Kalman Filter
        # THIS CAN BE TURNED OFF IF WRRANTED
        initial_encoder_state = kfs.update(encoder_state) 
        voState = kfs.update(voState)
        
        #initial_encoder_state = [initial_encoder_state[0], initial_encoder_state[1], ideal[2], initial_encoder_state[3], initial_encoder_state[4]]
        # Orientation override from IMU
        #voState = [voState[0], voState[1], voState[2], voState[3], voState[4]]
        #voState = [voState[0], voState[1], orientation[2], voState[3], voState[4]]
        
        # Call the sensor fusion function
        imu_override = orientation[2]
        adjusted_encoder_state = sf.sensor_fusion(timestep, voState, initial_encoder_state, imu_override)
        # ----- SENSOR FUSION -----

        # ----- START DWA HERE -----
        passvals = bot()

        # Add the sensor fusion outputs to the struct
        passvals.position[0] = adjusted_encoder_state[0]
        passvals.position[1] = adjusted_encoder_state[1]
        
        passvals.heading = adjusted_encoder_state[2]

        speed = adjusted_encoder_state[3]
        passvals.velocity[0] = speed*np.cos(passvals.heading)
        passvals.velocity[1] = speed*np.sin(passvals.heading)
        
        passvals.ang_vel = adjusted_encoder_state[4]


        # Lidar reading
        lidar.enablePointCloud()
        
        for RelativeCloudPoint in lidar.getPointCloud():
            x_ok = RelativeCloudPoint.x != np.inf and RelativeCloudPoint.x != -np.inf
            y_ok = RelativeCloudPoint.x != np.inf and RelativeCloudPoint.x != -np.inf
            zz = RelativeCloudPoint.x == 0 and RelativeCloudPoint.y == 0
            if x_ok and y_ok and not zz:
                x_idex = int(RelativeCloudPoint.x*3.1+15.5)
                y_idex = int(RelativeCloudPoint.y*3.1+15.5)
                
               
                #passvals.map.append([RelativeCloudPoint.y, RelativeCloudPoint.x])
                if 0 < x_idex and 30 > x_idex:
                    if 0 < y_idex and 30 > y_idex:
                        #print([x_idex,y_idex])
                        passvals.map[x_idex][y_idex] = 1
                        passvals.map[x_idex+1][y_idex] = 1
                        passvals.map[x_idex-1][y_idex] = 1
                        passvals.map[x_idex+1][y_idex+1] = 1
                        passvals.map[x_idex-1][y_idex-1] = 1
                        passvals.map[x_idex][y_idex+1] = 1
                        passvals.map[x_idex][y_idex-1] = 1
                        passvals.map[x_idex+1][y_idex-1] = 1
                        passvals.map[x_idex-1][y_idex+1] = 1


        vels = DWA(goal,passvals,0.4,0.5)
        desired_velocities = np.array([vels[1], vels[0]])
        if ~desired_velocities.any():
            goal = [30,48]
        # ----- END DWA HERE -----
                       
        current_state, error_state, wheel_vel_ideal = control.LQR(adjusted_encoder_state,error_state,desired_velocities)
        
        # THIS CODE HERE ISNT FOR THE WORKING OF THE ROBOT, IT'S BEING USED TO ANALYSE, ON THE FLY
        # THE VARIANCES BETWEEN THE IDEAL, ENCODER, VISUAL AND SENSOR FUSION STATES 
        # IT PRINTS TO THE CONS0LE PRODUCING THE GRAPHS FROM THE DISCORD CHAT
        #for j in range(5):
        #    print("----",i,"----")
        #    sheet1.write(i+1, j+1, ideal[j]) #Ideal State
        #    print("ID:", ideal)
        #    sheet1.write(i+1, j+6, encoder_state[j]) #Encoder State
        #    print("ED:", encoder_state)
        #    sheet1.write(i+1, j+11, voState[j]) #Visual Odometry State
        #    print("VO:", voState)
        #    sheet1.write(i+1, j+16, adjusted_encoder_state[j]) #Sensor Fusion State
        #    print("SF:", adjusted_encoder_state)
        #    a = ((adjusted_encoder_state[0] - ideal[0]))
        #    b = ((adjusted_encoder_state[1] - ideal[1]))
        #    c = ((adjusted_encoder_state[2] - ideal[2]))
        #    d = ((adjusted_encoder_state[3] - ideal[3]))
        #    e = ((adjusted_encoder_state[4] - ideal[4]))            
        #    print("VA:", [a, b, c, d, e], " | ")
        #    wb.save('xlwt example.xls')
 
        # ************* ENDS HERE 
       
        left_speed = wheel_vel_ideal[0]
        right_speed = wheel_vel_ideal[1]
        
        # ----- KALMAN FILTER -----
        #kf_left_speed, kf_right_speed = kf.update(left_speed, right_speed)
        #kf_left_speed = left_speed
        #kf_right_speed = right_speed
        # ----- KALMAN FILTER -----

        # ----- SENSOR FUSION -----
        #smoothed_left_speed, smoothed_right_speed = sf.smooth_velocity(left_speed, right_speed, prev_left_speed, prev_right_speed, alpha)
        #smoothed_left_speed = left_speed
        #smoothed_right_speed = right_speed
        
        # Max caps - Only used for testing purposes
        #if(smoothed_left_speed > 25):
        #    smoothed_left_speed = 25
        #if(smoothed_right_speed > 25):
        #    smoothed_right_speed = 25       
        
        # Seting smoothed values only
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        
        # Update the previous velocities
        prev_left_speed = left_speed
        prev_right_speed = right_speed
        # ----- SENSOR FUSION -----                

        #Update previous encoder information
        initial_encoder = copy.deepcopy(final_encoder)
        i += 1
    # Enter here exit cleanup code.
