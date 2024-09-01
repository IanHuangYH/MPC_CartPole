import casadi as ca
import numpy as np
import os
import control
import control.optimal as obc
import control as ct
import Model_NonlinearCartPole as CPM
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


folder_path = "/home/thuang/code/diffusion/MPC/CartPole/MPC_CartPole/data/nMPC"


############# parameter setting #####################
DEBUG = 1

# Time setting
TS = 0.01
H = 60
TIME_HOR = np.arange(0, H * TS, TS)
TOTAL_TIME = 3
t = np.arange(0, TOTAL_TIME, TS)

############### MPC setting ####################################
NUM_STATE = 5
TargetModel = ct.NonlinearIOSystem(
    CPM.dynamic_update_VirtualTheta, CPM.system_output, states=NUM_STATE, name='CartPole',
    inputs=('u'), outputs=('x', 'xdot', 'theta', 'thetadot','virtualtheta'))

# cost
Q_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
R = 0.01
P = np.diag([0.01, 1, 0, 10, 1000]) # get close to final point
P = np.diag([0, 0, 0, 0, 0]) 
Xf = np.zeros(NUM_STATE) # best final state
traj_cost = obc.quadratic_cost(TargetModel, Q, R, x0=Xf)
term_cost = obc.quadratic_cost(TargetModel, P, 0, x0=Xf)

# initial guess
u0 = 1000 # initial guess


# Define the initial states range
NUM_XINITIAL = 20
NUM_THETAINITIAL = 15
num_datagroup = NUM_XINITIAL * NUM_THETAINITIAL # number of data groups 


rng_x = np.linspace(-1,1,NUM_XINITIAL)
rng_theta = np.linspace(np.pi - np.pi/4, np.pi + np.pi/4,NUM_THETAINITIAL)

if DEBUG == 1:
    rng_theta = np.linspace(0 - np.pi/10, 0 + np.pi/10,NUM_THETAINITIAL)
    
rng0 = []
for m in rng_x:
    for n in rng_theta:
        rng0.append([m,n])
rng0 = np.array(rng0)


############# MPC Loop #####################
# data collecting loop
if DEBUG == 1:
    num_datagroup = 1
    
for turn in range(num_datagroup):
    # initial state
    x_0 = rng0[turn,0]
    #x_0= round(x_0, 3)
    theta_0 = rng0[turn,1]
    #theta_0= round(theta_0, 3)
    x0 = np.array([x_0, 0.0, theta_0, 0.0])  # Initial states
    if DEBUG == 1:
        x0 = np.array([0.0, 0.0, np.pi, 0.0, np.pi])
        #x0 = np.array([0.0, 0.0, np.pi+0.1, 0.0, 0.1**2/-np.pi+np.pi])

    # Simulation history
    x_hist = np.zeros((len(t), len(x0)))
    x_hist[0] = x0
    u_hist = np.zeros(len(t))
    cost_hist = np.zeros(len(t))

    # control loop
    for i in range(1, len(t)):
        # MPC horizon loop
        result = obc.solve_ocp(
            TargetModel, TIME_HOR, x0, traj_cost,
            terminal_cost=term_cost, initial_guess=u0)
        MPC_FirstU = result.inputs[0][0] #result.inputs[0] is 1xN array 
        dx = TargetModel.dynamics(t[i-1], x0, MPC_FirstU)
        x_hist[i] = x_hist[i-1] + dx*TS
        u_hist[i] = MPC_FirstU
        cost_hist[i] = result.cost

        # update
        x0 = x_hist[i]
        print(t[i])
        print("x0=",x_hist[i-1][0])
        print("x=",x_hist[i][0])
        print("x_dot=",x_hist[i][1])
        print("theta=",x_hist[i][2])
        print("theta_dot=",x_hist[i][3])
        print("u=",MPC_FirstU)
        print("u_hor=",result.inputs[0])
        print("simulation=",obc.OptimalControlResult(result.problem, result))
        print("----------------------------")
        


    ######################### save data ##########################
    # num_turn = turn + 1
    # num_turn_float = str(num_turn)

  
    # txtfile = 'initial states'
    # txt_name = txtfile + " " + num_turn_float + '.txt'
    # full_txt = os.path.join(folder_path, txt_name)
    # np.savetxt(full_txt, x0, delimiter=",",fmt='%1.3f')

    # # Save the control inputs to CSV files
    # cvsfile = 'u_data'
    # cvs_name = cvsfile + " " + num_turn_float + '.csv'
    # full_cvs = os.path.join(folder_path, cvs_name)
    # np.savetxt(full_cvs, U_sol, delimiter=",", fmt='%1.6f')

    ########################### plot some results #########################
    step = np.linspace(0,H,H+1)
    step_u = np.linspace(0,H-1,H)

    
    if turn in (0, 61, 134, 227, 295):
        plt.figure(figsize=(10, 8))
        plt.subplot(6, 1, 1)
        plt.plot(t, x_hist[:, 0])
        plt.ylabel('x (m)')
        plt.grid()
        plt.subplot(6, 1, 2)
        plt.plot(t, x_hist[:, 1])
        plt.ylabel('x_dot (m/s)')
        plt.grid()
        plt.subplot(6, 1, 3)
        plt.plot(t, x_hist[:, 2])
        plt.ylabel('theta (rad)')
        plt.grid()
        plt.subplot(6, 1, 4)
        plt.plot(t, x_hist[:, 3])
        plt.ylabel('theta_dot (rad/s)')
        plt.grid()
        plt.subplot(6, 1, 5)
        plt.plot(t, u_hist)
        plt.ylabel('u_value')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.subplot(6, 1, 6)
        plt.plot(t, cost_hist)
        plt.ylabel('cost')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.show()
        print("finished plot")

        # save plot 
        # plotfile = "plt"
        # plot_name = plotfile + " " + num_turn_float + '.png'
        # full_plot = os.path.join(folder_path, plot_name)
        # plt.savefig(full_plot)
