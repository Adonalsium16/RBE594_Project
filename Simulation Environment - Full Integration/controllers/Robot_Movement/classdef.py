import numpy as np

class bot:
    def __init__(self):
        # initialize everything to zero
        self.position = [0,0]
        self.heading  = 0 
        self.velocity = [0,0]
        self.ang_vel  = 0
        self.map      = np.zeros((31,31),dtype=bool)
        # map --> obstacle search space. 0 = no obstacles, 1 = obstacles
        # 3 m x 3 m grid with 0.01 m = 1 cm resolution 
        # 301 instead of 300 so that the bot can be considered at the exact center

        # Input:
        # * Obstacle Location (Relative to Robot) -> Camera && QR Code 
        # * Obstacle Type -> Camera 
        # * Robot Location (Relative to Map) -> Camera 
        # * Obstacle Location (Relative to Robot) -> LiDAR 
        # * Robot Heading (Actual) -> Camera (IMU)
        # * Robot Velocity (Actual) -> Encoder

        # Datatypes for these ^^ ?


        # Derivations:
        # - Robot Heading (Derived on Maths)
        # - Robot Location (Derived on Weighting)

        # Output:
        # * Structure Containing:
        # - Obstacle Location(s)
        # - Robot Location (Encoder Dervied)
        # - Robot Location (QR Code Derived)
        # - Robot Location (Weighted)
        # - Obstacle Type(s)
        # - Robot Velocity
        # - Robot Heading
    #def inflate(self):
    #    obs = np.nonzero(self.map)
    #    for ob in obs:



def example_func(b):
    #demonstrate pass by reference -- changes made inside functions propogate to the base
    b.position = [4,4]

if __name__ == "__main__":
    a = bot()
    print(a.position)
    print(a.heading)
    print(a.velocity)
    print(a.ang_vel)
    print(a.map)

    a.position = [3,5]

    print(a.position)

    example_func(a)
    print(a.position)


    a.map[0,0] = 1

    print(a.map)