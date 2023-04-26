import numpy as np
from classdef import bot


def create_DW(av_max,lv_max,n=100):
    av = np.linspace(-av_max, av_max, n)
    lv = np.linspace(0.01,lv_max,n)

    traj = np.zeros((n,n,2))

    for i in range(n):
        for j in range(n):
            traj[i][j][0] = av[i]
            traj[i][j][1] = lv[j]
    return traj

def filt_DW(DW,b):
    #print("Starting filter...")
    surv_traj = [[]]
    for i in range(len(DW)): # rows
        for j in range(len(DW[0])): #columns
            av = DW[i][j][0]
            lv = DW[i][j][1]
            
            if check_dynamics(av,lv,b):
                t = 0.1/lv
                #t = 1
                p = vels2traj(av,lv,b,t)
                #print(p[-1][0]-b.position[0])
                #print(p[-1][1]-b.position[1])
                if checkOccupancy(b,p):
                    #print("passed occupancy")
                    #print([av,lv])
                    if surv_traj == [[]]:
                        surv_traj = [[av,lv,p[-1][2]]]
                    else:
                        surv_traj.append([av,lv,p[-1][2]])
                #else: print("failed occupancy")
            #else: print("failed dynamics")
    #print(surv_traj)
    return surv_traj

def vels2traj(av,lv,b,t):
    heading = b.heading
    x = b.position[0]
    y = b.position[1]
    tstep = t/50
    for i in range(50):
        heading = heading + av*tstep
        x = x + lv*np.cos(heading)*tstep
        y = y + lv*np.sin(heading)*tstep
        if 'p' in locals():
            p.append([x,y,heading])
        else:
            p = [[x,y,heading]]

    #x_plt = [item[0] for item in p]
    #y_plt = [item[1] for item in p]
    #print(x_plt)
    #plt.scatter(x_plt, y_plt)
    #plt.show()
    return p

def checkOccupancy(b,p):
    map = b.map
    #print(np.nonzero(map))
    for pos in p:
        x = pos[0] - b.position[0]
        y = pos[1] - b.position[1]
        # x/y are in meters, map indices are in cm:
        #print(x)
        #print(y)
        #x_idex = int(x*10.333+15.5)
        #y_idex = int(y*10.333+15.5)
        x_idex = int(x*3.1+15.5)
        y_idex = int(y*3.1+15.5)
        #print(x_idex)
        #print(y_idex)
        if x_idex != 15 or y_idex != 15:
            if map[x_idex][y_idex]: #if we ever see an obstacle in the path, don't return the path as valid
                return False
    return True # if we made it through the whole path without obstacles, path is valid

def opt(paths,b,goal,max_vel=1.2):


    #print("CHOOSE BETWEEN:")

    best_score = np.inf
    #print(paths)
    for path in paths:
        traj = vels2traj(path[0],path[1],b,1)

        x_cur = traj[-1][0]
        y_cur = traj[-1][1]

        x_tar = goal[0]
        y_tar = goal[1]

        target_heading = np.arctan2(y_tar-y_cur,x_tar-x_cur)


        #print([x_cur,y_cur,x_tar,y_tar,target_heading,traj[-1][2]])

        #heading = np.abs(((b.heading+path[0]*1 - target_heading) + np.pi) % (2*np.pi) - np.pi)
        
        #heading = np.abs(((traj[-1][2] - target_heading) + np.pi) % (2*np.pi) - np.pi)
        
        # BOTH HEADING METHODS GIVE SAME RESULT
        #heading = ((target_heading - traj[-1][2]) + np.pi) % (2*np.pi) - np.pi

        heading = np.abs(np.arctan2(np.sin(target_heading - traj[-1][2]),np.cos(target_heading - traj[-1][2])))

        #print(target_heading)
        #print(b.heading+path[0]*0.1)
        #print(heading)
        vel = max_vel - path[1]
        
        dist = calc_obstacle_cost(traj,b)

        #print(path)
        #print([heading,vel,dist])

        score = heading + vel*5 + dist#*1000
        #print(score)
        if score < best_score:
            best = path
            best_score = score
            best_vel = vel
            best_dist = dist
            best_head = heading
    #print("CHOSEN:")
    #print(target_heading)
    #print(best_head)
    #print([best_head,best_vel,best_dist])
    #print(best_score)
    #print(best_vel)
    #print(best_dist)
    #print(best_head)
    return best

def calc_obstacle_cost(traj,state):
    map = state.map
    min_r = np.inf
    x_end = traj[-1][0]
    y_end = traj[-1][1]
    #print([x_end,y_end])
    if map.any():
        
        xy = np.nonzero(map)
        for i in range(len(xy[0])):
            #x = xy[0][i]/10 #obstacle x location in meters
            #y = xy[1][i]/10 #obstacle y location in meters
            x = (xy[0][i]-15.5)/3.1
            y = (xy[1][i]-15.5)/3.1
            #print([x,y])

            dx = x_end - (x+state.position[0])
            dy = y_end - (y+state.position[1])
            r = dx**2 + dy**2
            #print(r)
            if r < min_r:
                min_r = r
                

            #print([xy[0][i],xy[1][i]])
            #for pt in traj:
            ##    #print(traj) 
            #    #print(pt)
            #    dx = pt[0] - (x+state.position[0])
            #    dy = pt[1] - (y+state.position[1])
            #    r = dx**2 + dy**2
            #    #print(r)
            #    if r < min_r:
            #        min_r = r
            #        #print(r)
        #print(min_r)
        cost = 1/np.sqrt(min_r)
        #if min_r < 0.3:
        #    return np.inf
        return cost
    else: return 0


def check_dynamics(av,lv,b):
    vel = np.sqrt(b.velocity[0]**2 + b.velocity[1]**2)

    av_dev = np.absolute(b.ang_vel - av)
    lv_dev = np.absolute(vel - lv)
    if av_dev > 0.2:
        return False
    if lv_dev > 0.3:
        return False
    else:
        return True


def DWA(goal, b=bot(),av_max=1,lv_max=1.2):

    x_goal = goal[0]
    y_goal = goal[1]
    x_cur  = b.position[0]
    y_cur  = b.position[1]
    
    if np.sqrt((x_goal-x_cur)**2+(y_goal-y_cur)**2) < 1:
        return [0,0]

    DW = create_DW(av_max, lv_max,7) #keep this number odd so that 0 is an option
    valid_paths = filt_DW(DW,b)
    #print(len(valid_paths))
    best_vels = opt(valid_paths,b,goal,lv_max)
    #best_traj = vels2traj(best_vels[0],best_vels[1],b)

    return best_vels


if __name__ == "__main__":
    print(DWA([3,5]))