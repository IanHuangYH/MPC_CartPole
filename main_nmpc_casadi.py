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
NUM_STATE = 4
NUM_U = 1
H = 60 # prediction horizon
H_STATE = H + 1

# mpc parameters
Q = np.diag([10.0, 1.0, 0.0, 1.0])
R = 0.01
Q_redundant = 1000.0
Xf = np.array([0.0, 0.0, 0.0, 0.0]) # best final state
TS = 0.01
TIME_HOR = np.arange(0, H * TS, TS)
TOTAL_TIME = 3

t = np.arange(0, TOTAL_TIME, TS)
u_initial_value = 1000
u_update_control = u_initial_value * np.ones((NUM_U,H)) # initial guess
x_update_state = np.zeros((NUM_STATE,H_STATE))

# optimizor setting
# casadi_Opti
optimizer = ca.Opti()
X_state = optimizer.variable(4, H + 1) # 5xN+1
U_input = optimizer.variable(1, H) # 1xN


# Define the initial states range
NUM_XINITIAL = 20
NUM_THETAINITIAL = 15
num_datagroup = NUM_XINITIAL * NUM_THETAINITIAL # number of data groups 


rng_x = np.linspace(-1,1,NUM_XINITIAL)
rng_theta = np.linspace(np.pi - np.pi/4, np.pi + np.pi/4,NUM_THETAINITIAL)
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
        x0 = np.array([0.0, 0.0, np.pi, 0.0])
        
    x_update_state[:,0] = x0

    # Simulation history
    x_hist = np.zeros((len(t), len(x0)))
    x_hist[0] = x0
    u_hist = np.zeros(len(t))
    cost_hist = np.zeros(len(t))

    # optimizor
    optimizer.solver('ipopt')
    optimizer.subject_to(X_state[:, 0] == x0)  # Initial condition
    cost = 0
    for k in range(H):
        cost = cost + ca.mtimes([(X_state[:, k]-Xf).T, Q, (X_state[:, k]-Xf)])+ ca.mtimes([U_input[k], R, U_input[k]]) + ca.mtimes([-CPM.PI_UNDER_1*(X_state[2, k]-np.pi)**2+np.pi, Q_redundant, -CPM.PI_UNDER_1*(X_state[2, k]-np.pi)**2+np.pi])
        
        x_next = CPM.EulerForwardCartpole_Casadi(CPM.dynamic_update_Casadi,TS,X_state[:,k],U_input[k])
        optimizer.subject_to(X_state[:, k + 1] == x_next)
    
    optimizer.minimize(cost)

    
    # control loop
    for i in range(1, len(t)):
        optimizer.set_initial(U_input, u_update_control)
        optimizer.set_initial(X_state, x_update_state)
        sol = optimizer.solve()
        
        X_sol = sol.value(X_state)
        U_sol = sol.value(U_input)
        MPC_FirstU = U_sol[0]
        
        # record
        u_hist[i] = MPC_FirstU
        x_hist[i] = CPM.EulerForwardCartpole_Normal(TS,x_hist[i-1],MPC_FirstU)

        # update
        x_update_state = X_sol
        u_update_control = U_sol
        print(t[i])
        print("x0=",x_hist[i-1][0])
        print("x=",x_hist[i][0])
        print("x_dot=",x_hist[i][1])
        print("theta=",x_hist[i][2])
        print("theta_dot=",x_hist[i][3])
        print("u=",MPC_FirstU)
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
