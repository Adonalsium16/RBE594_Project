import math
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import random

np.set_printoptions(precision=3,suppress=True)

class Robot_Kinematics():

    def __init__(self,timestep):

        self.wheel_base = 0.6 # Distance between wheels in meters
        self.wheel_radius = 0.07 # Radius of wheels in meters

        self.dt = timestep/1000 # time step in seconds

    def orientation_bound(self,angle):
        
        if angle > np.pi:
            bounded_angle = angle - 2*np.pi
        elif angle < -np.pi:
            bounded_angle = angle + 2*np.pi
        else:
            bounded_angle = angle

        return bounded_angle
    
    def translation (self, current_state):
        """
        Calculates the translational change of the system
        based on the current state space parameters.

        :param current_state: Current state space
            State Space Form:   [X Position, Y Position, Angle  , Linear Velocity, Angular Velocity]
            State Space Units:  [Meters    , Meters    , Radians, Meters/second  , Radians/second  ]

        :return: New X and Y postions of the system
        """    
        
        postition = np.empty(2,dtype=object)

        postition[0] = current_state[0] + current_state[3] * math.cos(current_state[2]) * self.dt 
        postition[1] = current_state[1] + current_state[3] * math.sin(current_state[2]) * self.dt

        return postition
    
    def rotation (self, current_state):
        """
        Calculates the rotational change of the system
        based on the current state space parameters.

        :param current_state: Current state space
            State Space Form:   [X Position, Y Position, Angle  , Linear Velocity, Angular Velocity]
            State Space Units:  [Meters    , Meters    , Radians, Meters/second  , Radians/second  ]
            
        :return: New angle orientation of the system
        """

        angle = current_state[2] + current_state[4] * self.dt

        angle = self.orientation_bound(angle)

        return angle   
    
    def state_change(self,current_state,velocity_state):
        """
        Updates the state space based on the current state 
        of the system and the updated velocities

        :param current_state: Current state space
            State Space Form:   [X Position, Y Position, Angle  , Linear Velocity, Angular Velocity]
            State Space Units:  [Meters    , Meters    , Radians, Meters/second  , Radians/second  ]

        :param Velocity_state: Current state space
            Velocity State Form:   [Linear Velocity, Angular Velocity]
            Velocity State Units:  [Meters/second  , Radians/second  ]
            
        :return: New state space of the system
        """
        
        position = self.translation(current_state)
        angle = self.rotation(current_state)
        
        new_state = np.array([position[0],position[1],angle,velocity_state[0],velocity_state[1]])

        return new_state
    
    def wheel_speed_error(self,error_state,velocity_state):
        """
        Updates the state space based on the current state 
        of the system and the updated velocities. Estimated 
        error will be added to simulate encoder error  of 20 
        arcminutes and wheel slippage of up to 90 degree per 
        full wheel turn.

        :param error_state: Current state space with error
            State Space Form:   [X Position, Y Position, Angle  , Linear Velocity, Angular Velocity]
            State Space Units:  [Meters    , Meters    , Radians, Meters/second  , Radians/second  ]

        :param Velocity_state: desired velocities for the system
            Velocity State Form:   [Linear Velocity, Angular Velocity]
            Velocity State Units:  [Meters/second  , Radians/second  ]
            
        :return error_state: New state space of the system with error

        :return wheel_velocity_ideal: Ideal left/right wheel velocities

        :return wheel_velocity_err: Left/right wheel velocities with error
        """
        wheel_velocity_ideal = []

        wheel_ang_vel_r = (velocity_state[0] + velocity_state[1]*self.wheel_base/2)/self.wheel_radius
        wheel_ang_vel_l = (velocity_state[0] - velocity_state[1]*self.wheel_base/2)/self.wheel_radius
        
        wheel_velocity_ideal.append(wheel_ang_vel_l)
        wheel_velocity_ideal.append(wheel_ang_vel_r)

        error_state[0] = error_state[0] + error_state[3] * math.cos(error_state[2]) * self.dt
        error_state[1] = error_state[1] + error_state[3] * math.sin(error_state[2]) * self.dt
        error_state[2] = error_state[2] + error_state[4] * self.dt 
        error_state[2] = self.orientation_bound(error_state[2])
        error_state[3] = (wheel_ang_vel_r + wheel_ang_vel_l) * self.wheel_radius / 2
        error_state[4] = (wheel_ang_vel_r - wheel_ang_vel_l) * self.wheel_radius / self.wheel_base

        return error_state,wheel_velocity_ideal
    
    def encoder_state_estimate(self,current_state,encoder_diff_l,encoder_diff_r):

        encoder_state = [0.0,0.0,0.0,0.0,0.0]

        wheel_ang_vel_r = (encoder_diff_r) / self.dt
        wheel_ang_vel_l = (encoder_diff_l) / self.dt

        encoder_state[0] = current_state[0] + current_state[3] * math.cos(current_state[2]) * self.dt
        encoder_state[1] = current_state[1] + current_state[3] * math.sin(current_state[2]) * self.dt
        encoder_state[2] = current_state[2] + current_state[4] * self.dt 
        encoder_state[2] = self.orientation_bound(encoder_state[2])
        encoder_state[3] = (wheel_ang_vel_r + wheel_ang_vel_l) * self.wheel_radius / 2
        encoder_state[4] = (wheel_ang_vel_r - wheel_ang_vel_l) * self.wheel_radius / self.wheel_base

        return encoder_state
    
class Robot_Controller():
 
    def __init__(self,timestep):

        self.max_linear_acceleration = 0.5 # Acceleration in m/s^2
        self.max_anglular_acceleration = 0.05 # Acceleration in rads/s^2

        self.dt = timestep/1000 # time step in seconds

    def controller_lqr_discrete_time(self, A, B, Q, R):
        """
        Solve the discrete time LQR controller for a discrete time system.
        
        A and B are system matrices, describing the systems dynamics:
        x[k+1] = A x[k] + B u[k]
        
        The controller minimizes the infinite horizon quadratic cost function:
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        
        where Q is a positive semidefinite matrix, and R is positive definite matrix.
        
        Returns K, X, eigVals:
        Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
        The optimal input is then computed as:
        input: u = -K*x
        """

        #first, try to solve the ricatti equation
        X = scipy.linalg.solve_discrete_are(A, B, Q, R)
        
        #compute the LQR gain
        K = np.dot(np.linalg.inv(np.dot(np.dot(B.T,X),B)+R),(np.dot(np.dot(B.T,X),A)))  
        
        eigVals = np.linalg.eigvals(A-np.dot(B,K))
        
        return -K, X, eigVals

    def RegulateInput(self,control_input):
        """
        regulate delta to : - max_steer_angle ~ max_steer_angle
        regulate a to : - max_acceleration ~ max_acceleration

        :param max_anglular_acceleration: acceleration [rads/s^2]

        :param max_linear_acceleration: acceleration [m / s^2]

        :return: regulated linear and angular accelerations
        """      

        if control_input[1] < -1.0 * self.max_anglular_acceleration:
            control_input[1] = -1.0 * self.max_anglular_acceleration

        if control_input[1] > 1.0 * self.max_anglular_acceleration:
            control_input[1] = 1.0 * self.max_anglular_acceleration

        if control_input[0] < -1.0 * self.max_linear_acceleration:
            control_input[0] = -1.0 * self.max_linear_acceleration

        if control_input[0] > 1.0 * self.max_linear_acceleration:
            control_input[0] = 1.0 * self.max_linear_acceleration

        return control_input
    

    def LQR (self, system_state, system_state_err, desired_velocities):
        """
        Updates the state space based on the current state 
        of the system and the updated velocities

        :param current_state: Current state space
            State Space Form:   [X Position, Y Position, Angle  , Linear Velocity, Angular Velocity]
            State Space Units:  [Meters    , Meters    , Radians, Meters/second  , Radians/second  ]

        :param Velocity_state: Desired velocities for the system
            Velocity State Form:   [Linear Velocity, Angular Velocity]
            Velocity State Units:  [Meters/second  , Radians/second  ]
            
        :return system_state_f: New state space of the system

        :return system_state_err: New state space of the system with error
        
        :return wheel_velocity_ideal: Ideal left/right wheel velocities

        :return wheel_velocity_err: Left/right wheel velocities with error
        """

        A = np.array([  [1.0, 0],    
                        [0, 1.0]])

        B = np.array([  [self.dt,0],
                        [0,self.dt]])

        Q = np.array([  [1.0, 0],   # Penalize linear velocity error     
                        [0, 1.0]])  # Penalize angular velocity error

        R = np.array([  [1.0,0],    # Penalty for linear acceleration effort  
                        [0,1.0]])   # Penalty for angular acceleration effort

        K, X, eigVals = self.controller_lqr_discrete_time(A,B,Q,R)

        error_state = desired_velocities - system_state[3:5]

        control_state = np.matmul(K, error_state)

        control_state = self.RegulateInput(control_state)

        error_state_f = np.matmul(A,error_state) + np.matmul(B,control_state)

        velocity_state = desired_velocities - error_state_f

        system = Robot_Kinematics(self.dt)
        system_state_f = system.state_change(system_state,velocity_state)

        system_state_err,wheel_vel_ideal = system.wheel_speed_error(system_state_err,velocity_state)

        return system_state_f, system_state_err, wheel_vel_ideal

def Test_Run(run):
    """
    Test script to validate code
    """

    actual_state = np.array([0.0,0.0,0.0,0.0,0.0]) # State matrices need to change to a 5x1 matrix

    error_state = np.array([0.0,0.0,0.0,0.0,0.0]) # State matrices need to change to a 5x1 matrix

    desired_velocities = np.array([1.43,1.0])  # Velocity matrices need to change to a 2x1 matrix. Provided by the local planner (WP2 - Local Planner)    

    contoller = Robot_Controller(96)
    history_x = []
    history_y = []
    history_theta = []
    history_v = []
    history_w = []
    history_LW_vel = []
    history_RW_vel = []

    error_x = []
    error_y = []
    error_theta = []
    error_v = []
    error_w = []
    error_LW_vel = []
    error_RW_vel = []

    for i in range(2500):
        actual_state, error_state, wheel_vel_ideal = contoller.LQR(actual_state,error_state,desired_velocities)
        
        if i == 1250:
            desired_velocities = np.array([0,0])
            
        history_x.append(actual_state[0])
        history_y.append(actual_state[1])
        history_theta.append(actual_state[2])
        history_v.append(actual_state[3])
        history_w.append(actual_state[4])
        history_LW_vel.append(wheel_vel_ideal[0])
        history_RW_vel.append(wheel_vel_ideal[1])

        error_x.append(error_state[0])
        error_y.append(error_state[1])
        error_theta.append(error_state[2])
        error_v.append(error_state[3])
        error_w.append(error_state[4])
        #error_LW_vel.append(wheel_vel_err[0])
        #error_RW_vel.append(wheel_vel_err[1])

        print(actual_state-error_state)
    if run == 0:
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
        time = np.float32([*range(len(history_x))])*.1
        ax1.plot(time, history_x, label='x position')
        ax1.plot(time, error_x, label='x-err position')

        ax2.plot(time, history_y, label='y position')
        ax2.plot(time, error_y, label='y-err position')

        ax3.plot(time, history_theta, label='angle')
        ax3.plot(time, error_theta, label='angle-err')

        ax4.plot(time, history_v, label='linear velocity')
        ax4.plot(time, error_v, label='linear velocity-err')

        ax5.plot(time, history_w, label='angular velocity')
        ax5.plot(time, error_w, label='angular velocity-err')

        ax1.set_title("X Position")
        ax2.set_title("Y Position")
        ax3.set_title("Theta")
        ax4.set_title("Linear Velocity")
        ax5.set_title("Angular Velocity")
    if run == 1:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        time = np.float32([*range(len(history_x))])*.1
        ax1.plot(time, history_LW_vel, label='Left Wheel Velocity')
        #ax1.plot(time, error_LW_vel, label='Left Wheel Velocity with Error')

        ax2.plot(time, history_RW_vel, label='Right Wheel Velocity')
        #ax2.plot(time, error_RW_vel, label='Right Wheel Velocity with Error')

        ax1.set_title("Left Wheel Velocity")
        ax2.set_title("Right Wheel Velocity")
    
    plt.show()
    print('done')

#Test_Run(0)