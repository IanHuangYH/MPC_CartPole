import os
import casadi as ca
import numpy as np
import Model_NonlinearCartPole as CPM
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


folder_path = "data/NMPC/"


############# parameter setting #####################
DEBUG = 1
NUM_STATE = 5
NUM_U = 1
H = 60 # prediction horizon
H_STATE = H + 1

# mpc parameters
Q_redundant = 1000.0
P_redundant = 10000.0
Q = np.diag([0.01, 0.01, 0, 0.01, Q_redundant])
R = 0.001
P = np.diag([0.01, 1, 0, 1, P_redundant]) # get close to final point
P = np.diag([0, 0, 0, 0, 0]) 

TS = 0.01
TIME_HOR = np.arange(0, H * TS, TS)
TOTAL_TIME = 3

t = np.arange(0, TOTAL_TIME, TS)
u_initial_value = 10000
u_update_control = np.zeros((NUM_U,H)) # initial guess
u_update_control[0][0] = u_initial_value
x_update_state = np.zeros((NUM_STATE,H_STATE))

# pos
INITIAL_U = 1000
INITIAL_X = 10
# neg
INITIAL_U = -1000
INITIAL_X = 0
if INITIAL_U > 0:
    initial_guess_filename = 'pos'
else:
    initial_guess_filename = 'neg'
# optimizor setting
# casadi_Opti



#Xf = np.zeros(NUM_STATE) # best final state

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
    x0 = np.array([x_0, 0.0, theta_0, 0.0, np.pi])  # Initial states
    if DEBUG == 1:
        x0 = np.array([0.0, 0.0, np.pi, 0.0, np.pi])
        
    x_update_state[:,0] = x0

    # Simulation history
    x_hist = np.zeros((len(t), len(x0)))
    x_hist[0] = x0
    u_hist = np.zeros((len(t),1))
    cost_hist = np.zeros((len(t),1))

    
    # control loop
    for i in range(0, len(t)-1):
        optimizer = ca.Opti()
        X_state = optimizer.variable(NUM_STATE, H + 1) # 5xN+1
        U_input = optimizer.variable(1, H) # 1xN
        optimizer.set_initial(U_input, INITIAL_U)
        optimizer.set_initial(X_state, INITIAL_X)
        
        Xf = optimizer.parameter(NUM_STATE)
        optimizer.set_value(Xf, np.zeros(NUM_STATE))
        optimizer.subject_to(X_state[:, 0] == x0)  # Initial condition
        cost = 0
        cost += Q[0,0]*X_state[0, 0]**2 + Q[1,1]*X_state[1, 0]**2 + Q[2,2]*X_state[2, 0]**2 + Q[3,3]*X_state[3, 0]**2 + Q[4,4]*X_state[4, 0]**2
        for k in range(0,H-1):
            x_next = CPM.EulerForwardCartpole_virtual_Casadi(CPM.dynamic_update_virtual_Casadi,TS,X_state[:,k],U_input[k])
            optimizer.subject_to(X_state[:, k + 1] == x_next)
            cost += Q[0,0]*X_state[0, k+1]**2 + Q[1,1]*X_state[1, k+1]**2 + Q[2,2]*X_state[2, k+1]**2 + Q[3,3]*X_state[3, k+1]**2 + Q[4,4]*X_state[4, k+1]**2 + R*U_input[:, k]**2
        
        x_terminal = CPM.EulerForwardCartpole_virtual_Casadi(CPM.dynamic_update_virtual_Casadi,TS,X_state[:,H-1],U_input[H-1])
        optimizer.subject_to(X_state[:, H] == x_terminal)
        cost += P[0,0]*X_state[0, H]**2 + P[1,1]*X_state[1, H]**2 + P[2,2]*X_state[2, H]**2 + P[3,3]*X_state[3, H]**2 + P[4,4]*X_state[4, H]**2 + U_input[:, H-1]**2

        optimizer.minimize(cost)
        opts_setting = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        optimizer.solver('ipopt', opts_setting)
        sol = optimizer.solve()
        
        X_sol = sol.value(X_state)
        U_sol = sol.value(U_input)
        MPC_FirstU = U_sol[0]
        
        # update
        x_update_state = CPM.EulerForwardCartpole_virtual(TS,x_hist[i],MPC_FirstU)
        x0 = x_update_state
        u_update_control = U_sol
        #x0 = X_sol[:,1]
        
        # record
        u_hist[i][0] = MPC_FirstU
        #x_hist[i+1] = CPM.EulerForwardCartpole_virtual(TS,x_hist[i],MPC_FirstU)
        x_hist[i+1] = x0
        cost_hist[i+1][0] = sol.value(cost)

        print("t=", t[i])
        # print("x0=",x_hist[i][0])
        # print("x=",x_hist[i+1][0])
        # print("x_dot=",x_hist[i+1][1])
        # print("theta=",x_hist[i+1][2])
        # print("theta_dot=",x_hist[i+1][3])
        # print("theta_star=",x_hist[i+1][4])
        print("u=",MPC_FirstU)
        # print("cost=",cost_hist[i+1][0])
        # print("----------------------------")
        


    ######################### save data ##########################
    num_turn = turn + 1
    num_turn_float = str(num_turn)

    ini_state_txt = 'ini_states_'
    ini_guess_txt = 'ini_guess_'
    txt_name = ini_state_txt + num_turn_float + ini_guess_txt + initial_guess_filename + '.txt'
    full_txt = os.path.join(folder_path, txt_name)
    
    t_reshape = t.reshape(-1,1)
    result = np.hstack((t_reshape, x_hist, u_hist, cost_hist))
    np.savetxt(full_txt, result, fmt='%15.6f')

    ########################### plot some results #########################
    # step = np.linspace(0,H,H+1)
    # step_u = np.linspace(0,H-1,H)

    
    # if turn in (0, 61, 134, 227, 295):
    #     plt.figure(figsize=(10, 8))
    #     plt.subplot(7, 1, 1)
    #     plt.plot(t, x_hist[:, 0])
    #     plt.ylabel('x (m)')
    #     plt.grid()
    #     plt.subplot(7, 1, 2)
    #     plt.plot(t, x_hist[:, 1])
    #     plt.ylabel('x_dot (m/s)')
    #     plt.grid()
    #     plt.subplot(7, 1, 3)
    #     plt.plot(t, x_hist[:, 2])
    #     plt.ylabel('theta (rad)')
    #     plt.grid()
    #     plt.subplot(7, 1, 4)
    #     plt.plot(t, x_hist[:, 3])
    #     plt.ylabel('theta_dot (rad/s)')
    #     plt.grid()
    #     plt.subplot(7, 1, 5)
    #     plt.plot(t, x_hist[:, 4])
    #     plt.ylabel('theta_*')
    #     plt.xlabel('Time (s)')
    #     plt.grid()
    #     plt.subplot(7, 1, 6)
    #     plt.plot(t, u_hist)
    #     plt.ylabel('u_value')
    #     plt.xlabel('Time (s)')
    #     plt.grid()
    #     plt.subplot(7, 1, 7)
    #     plt.plot(t, cost_hist)
    #     plt.ylabel('cost')
    #     plt.xlabel('Time (s)')
    #     plt.grid()
    #     plt.show()
    #     print("finished plot")

        # save plot 
        # plotfile = "plt"
        # plot_name = plotfile + " " + num_turn_float + '.png'
        # full_plot = os.path.join(folder_path, plot_name)
        # plt.savefig(full_plot)
