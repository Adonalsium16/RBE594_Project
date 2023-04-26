# Import required libraries 
import time
import math
import Robot_Movement as rm
import numpy as np

# Define a function for sensor fusion
def sensor_fusion(timestep, visual_odometry, initial_encoder_state, imu_override):
    
# TO ADD IN INTERPOLATION AND EXTRAPOLATION BASED ON SAMPLE RATES*********************************    
    error_weighting = 1E-6
    # Extract the visual odometry measurements
    x_vo, y_vo, theta_vo, v_vo, w_vo = visual_odometry
    # Extract the initial encoder state measurements
    x_e, y_e, theta_e, v_e, w_e = initial_encoder_state
    
    # Calculate the time step
    delta_t = timestep  # replace with your actual time step
    
    # Calculate the change in position and orientation based on the visual odometry measurements
    #dx_vo = v_vo * math.cos(theta_vo) * delta_t
    #dy_vo = v_vo * math.sin(theta_vo) * delta_t
    #dtheta_vo = w_vo * delta_t
    
    # Update the position and orientation est mates based on the visual odometry measurements
    # ********************************************************************************************    
    # YOU ARE ABLE TO CHANGE THESE WEIGHTING (i.e., .25, .745, .75, .25, .05 and .95) TO WHATEVER
    # YOU LIKE, AT PRESENT, IT IS SHAPING AND WEIGHTING
    x_e_vo = ((((x_e)*1.0) + ((x_vo)*.0)))
    y_e_vo = ((((y_e)*1.0) + ((y_vo)*.0)))
    theta_e_vo = (theta_e*.5) + (theta_vo*.0) + (imu_override*.5)
    
    # Calculate the adjusted linear and angular velocities based on the visual odometry and initial encoder state measurements
    # ********************************************************************************************    
    # YOU ARE ABLE TO CHANGE THESE WEIGHTING (i.e., .25, .75, .25, .75) TO WHATEVER
    # YOU LIKE, AT PRESENT, IT IS SHAPING AND WEIGHTING
    v_adj = (v_vo*.0) + (v_e*1.0)
    w_adj = (w_vo*.0) + (w_e*1.0) 
    
    # Return the adjusted encoder state
    return [x_e_vo, y_e_vo, theta_e_vo, v_adj, w_adj]

class KalmanFilter:
    def __init__(self):
        self.x = np.array([[0], [0]])  # state vector
        self.P = np.diag([1000, 1000])  # covariance matrix

        self.A = np.array([[1, 0], [0, 1]])  # state transition matrix
        self.B = np.array([[1, 0], [0, 1]])  # input matrix
        self.H = np.array([[1, 0], [0, 1]])  # observation matrix
        self.Q = np.diag([0.01, 0.01])  # process noise covariance
        self.R = np.diag([0.001, 0.001])  # measurement noise covariance

    def update(self, left_speed, right_speed):
        u = np.array([[left_speed], [right_speed]])  # input vector
        # predict
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        # update
        y = np.array([[left_speed], [right_speed]]) - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(2) - np.dot(K, self.H), self.P)
        return self.x[0][0], self.x[1][0]

# Define the velocity smoothing function
def smooth_velocity(left_speed, right_speed, prev_left_speed, prev_right_speed, alpha):
    # Compute the smoothed velocity
    left_speed = alpha * left_speed + (1 - alpha) * prev_left_speed
    right_speed = alpha * right_speed + (1 - alpha) * prev_right_speed
    
    # Return the updated velocities
    return left_speed, right_speed

class KalmanFilter_State: #Only Used For Visual Odometry
    def __init__(self):
        # Initialize the Kalman filter with some initial values
        # YOU ARE ABLE TO CHANGE THESE WEIGHTING (i.e., .995) TO WHATEVER
        # YOU LIKE, AT PRESENT, IT IS SHAPING AND WEIGHTING
        dt = 1.0
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # Initial state estimate
        self.P = np.eye(5) * 0 # Initial state estimate error covariance
        
        # YOU ARE ABLE TO CHANGE thje DIAGONAL VALUES TO WHATEVER
        # YOU LIKE, AT PRESENT, IT IS SHAPING AND WEIGHTING FOR BOTH MATRICES AND THE COVARIANCE
        self.F = np.array([[1, 0.0, 0.0, dt, 0.0],
                           [0.0, 1.0, 0.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0]]) # State transition matrix
        self.Q = np.eye(5) * 1.0 # Process noise covariance
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1]]) # Measurement matrix
        self.R = np.eye(5) * 0 # Measurement noise covariance

    def update(self, z):
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(5) - K @ self.H) @ P_pred

        # Return the estimated state
        return self.x
