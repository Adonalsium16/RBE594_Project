"""Robot_Movement controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import Dyanamics_and_Controls as dc
import Global_Planner as gp
import VisOdom_integration as VisOdom
import numpy as np
import Sensor_Fusion as sf
import copy
from controller import Robot, Motor, PositionSensor
from numpy import mat, eye, zeros, array
from numpy.linalg import inv

if __name__ == "__main__":

    # create the Robot instance.
    robot = Robot()
    
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
  
    # Define Kalman filter parameters
    A = mat([[1, 0], [0, 1]])  # state transition matrix
    B = mat([[0.5, 0.5], [0.5, 0.5]])  # input matrix
    C = mat([[1, 0], [0, 1]])  # output matrix
    Q = mat([[0.01, 0], [0, 0.01]])  # process noise covariance
    R = mat([[0.05, 0], [0, 0.05]])  # measurement noise covariance
    P = mat([[0.1, 0], [0, 0.1]])  # initial estimation error covariance
    x_hat = mat([[0], [0], [0], [0]])  # initial state estimation

    # Initialize variables
    last_time = 0
    last_left_position = left_encoder.getValue()
    last_right_position = right_encoder.getValue()
    last_left_velocity = 0
    last_right_velocity = 0
      
    i = 0
      
    vo = VisOdom.VisualOdom(left_camera)
    
    start7267,start2288,start2346,start6848,start6702 = gp.global_planner()
    
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        #Initialization of state based on current values
        if i == 0:
            orientation = imu.getRollPitchYaw()
            position = gps.getValues()
            speed = gps.getSpeed()
            ang_vel = gyro.getValues()
            actual_state[0] = position[0]
            actual_state[1] = position[1]
            actual_state[2] = orientation[2]
            actual_state[3] = speed
            actual_state[4] = ang_vel[2]
            ps_values[0]=left_ps.getValue()
            ps_values[1]=right_ps.getValue()
            initial_encoder = ps_values
            #print(initial_encoder)
            i += 1

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

        current_time = robot.getTime()
        current_left_position = left_ps.getValue()
        current_right_position = right_ps.getValue()
        current_left_velocity = (current_left_position - last_left_position) / (current_time - last_time)
        current_right_velocity = (current_right_position - last_right_position) / (current_time - last_time)

        # Get desired wheel velocities as input
        desired_left_velocity = 2.0  # replace with your own desired velocity
        desired_right_velocity = 2.0  # replace with your own desired velocity
        
        # Apply Kalman filter to left wheel velocity
        y = mat([[current_left_velocity], [current_right_velocity]])
        u = mat([[desired_left_velocity], [desired_right_velocity]])
        x_minus = A * x_hat + B * u
        P_minus = A * P * A.transpose() + Q
        K = P_minus * C.transpose() * inv(C * P_minus * C.transpose() + R)
        x_hat = x_minus + K * (y - C * x_minus)
        P = (eye(4) - K * C) * P_minus
        left_velocity = x_hat[0, 0]
        
        # Apply Kalman filter to right wheel velocity
        y = mat([[current_right_velocity], [current_left_velocity]])
        u = mat([[desired_right_velocity], [desired_left_velocity]])
        x_minus = A * x_hat + B * u
        P_minus = A * P * A.transpose() + Q
        K = P_minus * C.transpose() * inv(C * P_minus * C.transpose() + R)
        x_hat = x_minus + K * (y - C * x_minus)
        P = (eye(4) - K * C) * P_minus
        right_velocity = x_hat[1, 0]      
        
        # Set motor velocities based on Kalman filter output
        left_motor.setVelocity(left_velocity)
        right_motor.setVelocity(right_velocity)
        
        # Update variables for next iteration
        last_time = current_time
        last_left_position = current_left_position
        last_right_position = current_right_position
        last_left_velocity = left_velocity
        last_right_velocity = right_velocity  
        
    # Enter here exit cleanup code.
