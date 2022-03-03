import os
import numpy as np
import pandas as pd
from pandas import read_csv
import time
import csv
import tqdm
import random
from scipy.integrate import odeint, ode,solve_ivp
from joblib import Parallel, delayed
import multiprocessing
import pdb
import matplotlib.pyplot as plt
import argparse
# import h5py
from sympy import symbols, Eq, solve
import math


parser = argparse.ArgumentParser()
parser.add_argument("--case", type=int, 
    default="0", help="1 or 2 or 3 or 4 or 5 or 6 or 7 or 8")
parser.add_argument('--num_cores', type=int,
    default=2)
parser.add_argument('--K', type=int,
    default=50000000,help = "Carrying capacity")
parser.add_argument('--A_add', type=int,
    default=1000000, help = "Initial value of aphid population")
parser.add_argument('--Ratio', type=int,
    default=200, help = "A_add/H_add")
parser.add_argument('--alter', type=str,
    default="aw00", help="..")
parser.add_argument('--scenario', type=str,
     default="eco", help="f5")
parser.add_argument('--start_idx', type=int,
    default=0)
parser.add_argument('--end_idx', type=int,
    default=400)

# parser.add_argument("--species", type=str, 
#     default="aphid", help="aphid or ladybird")
# parser.add_argument("--data_dir", type=str, 
#     default="data_dump", help="Directory to the data folder!")

args = parser.parse_args()

case = args.case
Ratio = args.Ratio
alter = args.alter

#processor
num_cores = args.num_cores 
# species = args.species


def draw_multi_lines(x,pltt,folder,a,w,Tmin,Tmax,xlabel,ylabel):
    for i in range(x.shape[0]):
        x_ = x[i]
        plt.plot(x_)
    plt.title(pltt + ': a = {} w = {} Tmin = {} Tmax = {}'.format(a,w,Tmin,Tmax))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fn = 'a{}_w{}_{}_{}.png'.format(a,w,Tmin,Tmax)
    out_file = os.path.join(folder, fn)
    if not os.path.isfile(out_file):
        plt.savefig(out_file,bbox_inches='tight')
    plt.close()

def draw_multi_scatters(x,y,folder,a,w,Tmin,Tmax,xlabel,ylabel):
    for i in range(x.shape[0]):
        x_ = x[i]
        y_ = y[i]
        plt.scatter(x_,y_)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('AL : a = {} w = {} Tmin = {} Tmax = {}'.format(a,w,Tmin,Tmax))
    fn = 'a{}_w{}_{}_{}.png'.format(a,w,Tmin,Tmax)
    out_file = os.path.join(folder, fn)
    if not os.path.isfile(out_file):
        plt.savefig(out_file,bbox_inches='tight')
    plt.close()

def writer_csv(rows, filename):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def get_paras_TPC(Tmin,Tmax,Topt,q1):
    q2 = (q1*Tmax-q1*Topt)/(Topt -Tmin)
    return q2

def get_paras_mor2(Tmin,Tmax,Topt1,Topt2,v_max,v_min):
    # tic = time.time()
    a1,a2,b1,b2 = symbols('a1 a2 b1 b2')
    eq1 = Eq(a1*Tmin + b1, v_max)
    eq2 = Eq(a1*Topt1 + b1, v_min)
    eq3 = Eq(a2*Tmax + b2, v_max)
    eq4 = Eq(a2*Topt2 + b2, v_min)
    roots = solve((eq1,eq2,eq3,eq4),(a1,a2,b1,b2))
    a1 = roots.get(a1)
    a2 = roots.get(a2)
    b1 = roots.get(b1)
    b2 = roots.get(b2)
    # print(f"eta: {time.time() - tic}")
    return a1,a2,b1,b2

def mor_rate(T, a1,a2,b1,b2, Tmin,Tmax,Topt1,Topt2,v_max,v_min):
    # the function of temperature dependent mortality rate
    if T >= Tmin and T <= Topt1:
        rate = a1*T + b1
    elif T > Topt1 and T < Topt2:
        rate = v_min
    elif T >= Topt2 and T <= Tmax:
        rate = a2*T + b2
    else: rate = v_max
    return rate

def temp_t(a,w,t):
    y = -a*math.cos(2*math.pi*t/365) + w
    return y

def temps(a,w):
    x = np.arange(365)
    y = []
    for x_ in x:
        y_ = -a*math.cos(2*math.pi*x_/365) + w
        y.append(y_)
    y = np.array(y)
    return y

def batch(a,w,export_fns,year):
    tic = time.time()
    export_folder, plotAnp_folder,plotAp_folder, plotL_folder, plotAL_folder = export_fns   
    
    #output file name
    temp_fn = "a_{}_w_{}.csv".format(a,w)
    out_file = os.path.join(export_folder, temp_fn)
    if os.path.exists(out_file):
        print('{} exits!!!'.format(temp_fn))
        return False

    # PARAMETER VALUES FOR APHID
    m_devA = 0.9421
    m_fecA = 4.848
    q1_devA = 1.5
    q1_fecA = 1.5
    Topt_devA = 25
    Topt_fecA = 25
    Topt_morA1 = 20
    Topt_morA2 = 30
    v_maxA = 0.3
    v_minA = 0.03
    q2_devA = get_paras_TPC(TminA,TmaxA,Topt_devA,q1_devA)
    q2_fecA = get_paras_TPC(TminA,TmaxA,Topt_fecA,q1_fecA)
    # a0A, a1A,a2A,a3A = get_paras_mor(TminA,TmaxA,Topt_devA,v_maxA,v_minA)
    a1A, a2A, b1A, b2A = get_paras_mor2(TminA,TmaxA,Topt_morA1,Topt_morA2,v_maxA,v_minA)

    # PARAMETER VALUES FOR LADYBIRD
    # Development
    m_devL = 0.326 
    q1_devL = 1.5 
    Topt_devL = 25
    Topt_morL1 = 20
    Topt_morL2 = 30
    #coefficients for tdpr equations
    m_tdpr = 1
    q1_tdpr = 1.5
    Topt_tdpr = 25

    q2_devL = get_paras_TPC(TminL,TmaxL,Topt_devL,q1_devL)
    q2_tdpr = get_paras_TPC(TminL,TmaxL,Topt_tdpr,q1_tdpr)
    # mortality rate cofficients
    v_maxL = 0.15
    v_minL = 0.01
    a1L, a2L, b1L, b2L = get_paras_mor2(TminL,TmaxL,Topt_morL1,Topt_morL2,v_maxL,v_minL)

    # Other parameters
    K = args.K #carrying cacpacity of ladybird
    Qp = 100 # transformation rate of ladybird
    # v_max = 0.3 # maximal mortality rate for aphid and ladybird
    theta = 0.5 # ratio of female ladybird to male ladybird

    A_add = args.A_add
 
    def indicator(t,Tmin,Tmax):
        # Tmin = Tmin_fap or TminL
        # Tmax = TmaxL or TmaxL
        i = int(np.floor(t))
        if (t >= 5) & (np.all(temp[i-5:i+1] >= Tmin)) & (np.all(temp[i-5:i+1] <= Tmax)):
                return True
        else: return False

    def thorneley_france(m,Tmin,Tmax,Topt,q1,q2,T):
        # Generic temperature-dependent function
        if (T>=Tmin) & (T<=Tmax):
            return m*(((T-Tmin)**q1)*((Tmax-T)**q2))/(((Topt-Tmin)**q1)*((Tmax-Topt)**q2))
        else: return 0

    def polynomial(a0,a1,a2,a3,Tmin,Tmax,T,v_max):
        if (T>=Tmin) & (T<=Tmax):
            return np.minimum(a0+a1*T+a2*T**2+a3*T**3,v_max)
        else: return v_max

    def fdpr(a,Th,Aden_t):
        #food_dependent_predation_rate
        return a*Aden_t/(1+a*Th*Aden_t)

    def carring_capacity(Aden, K):
        return 1 - Aden/K


    #Model equations:
    def Solve_euler_model(var0,t_start,t_end,dt,predation):
        if predation == True:
            a_fac = 0.000001
        else:
            a_fac = 0
        #predation rate 
        #a_fac = 0.00001
        Th_fac = 1
        a1 = 1.464*a_fac
        Th1 = 0.01613*Th_fac
        #max = 1/Th  = 61.99 per day per ladybird
        a2 = 1.177*a_fac
        Th2 = 0.008982*Th_fac
        #max = 1/Th  = 111.33 per day per ladybird
        a3 = 1.437*a_fac
        Th3 = 0.01155*Th_fac
        #max = 1/Th  = 86.58 per day per ladybird
        a4 = 1.219*a_fac
        Th4 = 0.003985*Th_fac
        #max = 1/Th  = 250.9
        af = 1.461*a_fac
        Thf = 0.004453*Th_fac
        #max = 1/Th  = 224.56
        am = 1.461*a_fac
        Thm = 0.004453*Th_fac


        def fecL(Aden,Temp_t,tdpr_t):
            # Fecundity rate of female ladybirds
            if Temp_t <= TmaxL:
                return tdpr_t*fdpr(af, Thf, Aden)/Qp
            else:
                return 0

        ts = np.arange(t_start,t_end,dt)
        n_t=len(ts)
        Aap = np.zeros([n_t]);A1 = np.zeros([n_t]); A2 = np.zeros([n_t]); A3 = np.zeros([n_t]); A4 = np.zeros([n_t])
        Legg = np.zeros([n_t]); L1 = np.zeros([n_t]); L2 = np.zeros([n_t]); L3 = np.zeros([n_t]); L4 = np.zeros([n_t]); Lpupa = np.zeros([n_t]); Lf = np.zeros([n_t]); Lm = np.zeros([n_t])
        Aden = np.zeros([n_t]); Lden = np.zeros([n_t]); Predated_prey = np.zeros([n_t]); Adeath = np.zeros([n_t])
        Aap[0],A1[0],A2[0],A3[0],A4[0],Legg[0],L1[0],L2[0],L3[0],L4[0],Lpupa[0],Lf[0],Lm[0] = var0
        Aden[0] = Aap[0] + A1[0] + A2[0] + A3[0] + A4[0]
        Lden[0] = Legg[0] + L1[0] + L2[0] + L3[0] + L4[0] + Lpupa[0] + Lf[0] + Lm[0]
        
        num_change_A = 0; num_change_L = 0
        for i in range(1, n_t):
            for j in np.arange(2):
                t = ts[i-1] #previous time step
                t_cur = ts[i] #current time step
                if num_change_A == 0 and num_change_L == 0:
                    Aden_t, Aap_t, A1_t, A2_t, A3_t, A4_t = [A_add,A_add,0,0,0,0]
                    Lden_t, Legg_t, L1_t, L2_t, L3_t, L4_t, Lpupa_t, Lf_t, Lm_t = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif num_change_A == 1 and num_change_L == 0:
                    Aden_t = Aden[i-1];Aap_t = Aap[i-1]; A1_t = A1[i-1]; A2_t = A2[i-1]; A3_t = A3[i-1]; A4_t = A4[i-1]
                    if j == 0:
                        Lden_t, Legg_t, L1_t, L2_t, L3_t, L4_t, Lpupa_t, Lf_t, Lm_t = [A_add/Ratio, 0, 0, 0, 0, 0, 0, A_add/Ratio, 0]
                    else:
                        Lden_t, Legg_t, L1_t, L2_t, L3_t, L4_t, Lpupa_t, Lf_t, Lm_t = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif num_change_A == 1 and num_change_L == 1:
                    Aap_t = Aap[i-1]; A1_t = A1[i-1]; A2_t = A2[i-1]; A3_t = A3[i-1]; A4_t = A4[i-1]; Aden_t = Aden[i-1]
                    Legg_t = Legg[i-1]; L1_t = L1[i-1]; L2_t = L2[i-1]; L3_t = L3[i-1]; L4_t = L4[i-1]; Lpupa_t = Lpupa[i-1]; Lf_t = Lf[i-1]; Lm_t = Lm[i-1]; Lden_t = Lden[i-1]
                
                #integrated temperature
                # Temperature at t:
                Temp_t = temp_t(a,w,t)

                ## Temperature-dependent parameters of aphid
                #fecudity rate
                f_ap_t = thorneley_france(m_fecA, TminA, TmaxA, Topt_fecA, q1_fecA, q2_fecA, Temp_t)
                # development
                varphi_t = thorneley_france(m_devA, TminA, TmaxA, Topt_devA, q1_devA, q2_devA, Temp_t)
                # mortality
                mu_inst_t = mor_rate(Temp_t, a1A, a2A, b1A, b2A, TminA,TmaxA,Topt_morA1,Topt_morA2,v_maxA,v_minA)
                mu_ap_t = mu_inst_t
                #carring capacity
                k_effect_t = carring_capacity(Aden_t,K)

                ## Temperature-dependent parameters of ladybird
                #Predartion rate as a function of temperature at maximal aphid density
                #temperature dependent predation rate
                tdpr_t = thorneley_france(m_tdpr,TminL,TmaxL,Topt_tdpr,q1_tdpr,q2_tdpr,Temp_t)
                # Fecundity rate of female ladybirds
                f_L_t = fecL(Aden_t,Temp_t,tdpr_t) 
                #Stage-specific Development rates
                #Temperature-dependent development rates for egg and pupa
                delta_egg_t = thorneley_france(m_devL, TminL, TmaxL, Topt_devL, q1_devL, q2_devL, Temp_t)
                delta_pupa_t = delta_egg_t
                #Temperature-dependent developments rates at prey saturation
                delta_inst1_prey_saturation_t = delta_egg_t
                delta_inst2_prey_saturation_t = delta_egg_t
                delta_inst3_prey_saturation_t = delta_egg_t
                delta_inst4_prey_saturation_t = delta_egg_t
                # Mortality rate of various stages
                gamma_egg_t = mor_rate(Temp_t, a1L,a2L,b1L,b2L,TminL,TmaxL,Topt_morL1,Topt_morL2,v_maxL,v_minL)
                gamma_inst1_t = gamma_egg_t
                gamma_inst2_t = gamma_egg_t
                gamma_inst3_t = gamma_egg_t
                gamma_inst4_t = gamma_egg_t
                gamma_pupa_t = gamma_egg_t 
                gamma_f_t = gamma_egg_t
                gamma_m_t = gamma_egg_t
                # common parameters for dA_dt
                common_pA_t = tdpr_t*(L1_t*a1/(1+a1*Th1*Aden_t) + L2_t*a2/(1+a2*Th2*Aden_t) + L3_t*a3/(1+a3*Th3*Aden_t) + L4_t*a4/(1+a4*Th4*Aden_t) + Lf_t*af/(1+af*Thf*Aden_t) + Lm_t*am/(1+am*Thm*Aden_t))
                dA1_dt = f_ap_t*k_effect_t*Aap_t - mu_inst_t*A1_t - varphi_t*A1_t - A1_t*common_pA_t
                dA2_dt = varphi_t*A1_t - mu_inst_t*A2_t - varphi_t*A2_t - A2_t*common_pA_t
                dA3_dt = varphi_t*A2_t - mu_inst_t*A3_t - varphi_t*A3_t - A3_t*common_pA_t
                dA4_dt = varphi_t*A3_t - mu_inst_t*A4_t - varphi_t*A4_t - A4_t*common_pA_t
                dAap_dt = varphi_t*A4_t - mu_ap_t*Aap_t - Aap_t*common_pA_t

                #parameters for dL_dt
                delta_L1_t = delta_inst1_prey_saturation_t*tdpr_t*fdpr(a1,Th1,Aden_t)*Th1
                delta_L2_t = delta_inst2_prey_saturation_t*tdpr_t*fdpr(a2,Th2,Aden_t)*Th2
                delta_L3_t = delta_inst3_prey_saturation_t*tdpr_t*fdpr(a3,Th3,Aden_t)*Th3
                delta_L4_t = delta_inst4_prey_saturation_t*tdpr_t*fdpr(a4,Th4,Aden_t)*Th4

                dLegg_dt = f_L_t*Lf_t - (gamma_egg_t + delta_egg_t)*Legg_t
                dL1_dt = delta_egg_t*Legg_t - delta_L1_t*L1_t - gamma_inst1_t*L1_t
                dL2_dt = delta_L1_t*L1_t - delta_L2_t*L2_t - gamma_inst2_t*L2_t
                dL3_dt = delta_L2_t*L2_t - delta_L3_t*L3_t - gamma_inst3_t*L3_t
                dL4_dt = delta_L3_t*L3_t - delta_L4_t*L4_t - gamma_inst4_t*L4_t
                dLpupa_dt = delta_L4_t*L4_t - delta_pupa_t*Lpupa_t - gamma_pupa_t*Lpupa_t
                dLf_dt = theta*delta_pupa_t*Lpupa_t - gamma_f_t*Lf_t
                dLm_dt = (1-theta)*delta_pupa_t*Lpupa_t - gamma_m_t*Lm_t

                dAL_dt = [dAap_dt,dA1_dt,dA1_dt,dA2_dt,dA3_dt,dA4_dt,dLegg_dt,dL1_dt,dL2_dt,dL3_dt,dL4_dt,dLpupa_dt,dLf_dt,dLm_dt]
                Predated_prey[i] = common_pA_t*(Aap_t + A1_t + A2_t + A3_t + A4_t)
                Adeath[i] = mu_ap_t*Aap_t + mu_inst_t*A1_t + mu_inst_t*A2_t + mu_inst_t*A3_t + mu_inst_t*A4_t

                Aap[i] = dt*dAap_dt + Aap_t
                A1[i] =dt*dA1_dt + A1_t
                A2[i] = dt*dA2_dt + A2_t
                A3[i] = dt*dA3_dt + A3_t
                A4[i] = dt*dA4_dt + A4_t
                Aden[i] = Aap[i] + A1[i] + A2[i] + A3[i] + A4[i]
                
                Legg[i] = dt*dLegg_dt + Legg_t
                L1[i] = dt*dL1_dt + L1_t
                L2[i] = dt*dL2_dt + L2_t
                L3[i] = dt*dL3_dt + L3_t
                L4[i] = dt*dL4_dt + L4_t
                Lpupa[i] = dt*dLpupa_dt + Lpupa_t
                Lf[i] = dt*dLf_dt + Lf_t
                Lm[i] = dt*dLm_dt + Lm_t
                Lden[i] = Legg[i] +  L1[i] + L2[i] + L3[i] + L4[i] +  Lpupa[i] + Lf[i] + Lm[i]

                if Aap[i] < 0: Aap[i] = 0
                if A1[i] < 0: A1[i] = 0
                if A2[i] < 0: A2[i] = 0
                if A3[i] < 0: A3[i] = 0
                if A4[i] < 0: A4[i] = 0
                if Aden[i] <1: Aden[i] = 0;Aap[i] = 0; A1[i] = 0; A2[i] = 0; A3[i] = 0; A4[i] = 0

                if Legg[i] < 0: Legg[i] = 0
                if L1[i] < 0: L1[i] = 0
                if L2[i] < 0: L2[i] = 0
                if L3[i] < 0: L3[i] = 0
                if L4[i] < 0: L4[i] = 0
                if Lpupa[i] < 0: Lpupa[i] = 0
                if Lf[i] < 0: Lf[i] = 0
                if Lm[i] < 0: Lm[i] = 0
                if Lden[i] < 1: Lden[i] = 0;Legg[i] = 0; L1[i] = 0; L2[i] = 0; L3[i] = 0; L4[i] = 0; Lpupa[i] = 0; Lf[i] = 0; Lm[i] = 0

                # set start condition
                if Aden[i] <= Aden_t and num_change_A == 0 and j == 0:
                    Aden[i] = 0; Aap[i] = 0; A1[i] = 0; A2[i] = 0; A3[i] = 0; A4[i] = 0;Adeath[i] = 0
                    break
                if Aden[i] > Aden_t and num_change_A == 0 and j == 0:
                    num_change_A = 1
                    break
                if num_change_A == 1 and num_change_L == 0 and Lden[i] > Lden_t and j == 0:
                    num_change_L = 1
                    #import pdb;pdb.set_trace()
                    break
                else: continue
                if num_change_A == 1 and num_change_L == 1:
                    break
            if Temp_t >= 10 and num_change_A == 1 and num_change_L == 1:
                if Aden[i] == 0:
                    Aden = np.zeros([n_t]); Aap = np.zeros([n_t]);A1 = np.zeros([n_t]); A2 = np.zeros([n_t]); A3 = np.zeros([n_t]); A4 = np.zeros([n_t])
                    break
            elif Temp_t < 10 and num_change_A == 1 and num_change_L == 1:
                break
            Pre_AL = [Aap_t,A1_t,A2_t,A3_t,A4_t,Legg_t,L1_t,L2_t,L3_t,L4_t,Lpupa_t,Lf_t,Lm_t]
            Cur_AL = [Aap[i],A1[i],A2[i],A3[i],A4[i],Legg[i],L1[i],L2[i],L3[i],L4[i],Lpupa[i],Lf[i],Lm[i]]
        outputs = [Aden, Lden, Predated_prey, Adeath, Aap, A1, A2, A3, A4, Legg, L1, L2, L3, L4, Lpupa, Lf, Lm]
        return outputs

    years = np.arange(0,1)
    Adens_np = [];Adens_p = [];Ldens_p = [];Predated_preys = [];Adeaths = []
    Adens_np_day = [];Adens_p_day =[];Ldens_p_day = [];Predated_preys_day = []; Adeaths_day = []
    for year in years:
        #print(year,flush=True)
        # daily temperatures
        temp = temps(a,w)
        var0 = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        #model outputs (with predation and no predation)
        outputs_p = Solve_euler_model(var0,t_start = 0, t_end = 365,dt=0.01,predation = True)
        outputs_np = Solve_euler_model(var0,t_start = 0, t_end = 365,dt=0.01,predation = False)
        Aden_p = outputs_p[0]; Lden_p = outputs_p[1]; Predated_prey = outputs_p[2]; Adeath = outputs_p[3]
        Aden_np = outputs_np[0]; Lden_np = outputs_np[1]
        #daily outputs
        Aden_np_day = Aden_np[::100]
        Aden_p_day = Aden_p[::100]
        Lden_p_day = Lden_p[::100]
        Predated_prey_day = Predated_prey[::100]
        Adeath_day = Adeath[::100]
        Adens_np_day += [Aden_np_day]
        Adens_p_day += [Aden_p_day]
        Ldens_p_day += [Lden_p_day]
        Predated_preys_day += [Predated_prey_day]
        Adeaths_day += [Adeath_day]
    Adens_np_day = np.array(Adens_np_day)
    Adens_p_day = np.array(Adens_p_day)
    Ldens_p_day = np.array(Ldens_p_day)
    Predated_preys_day = np.array(Predated_preys_day)
    Adeaths_day = np.array(Adeaths_day)
    AL = np.concatenate((Adens_np_day,Adens_p_day,Ldens_p_day,Predated_preys_day,Adeaths_day))
    AL = pd.DataFrame(np.transpose(AL))
    if not os.path.isfile(out_file):
        AL.to_csv(out_file,header=None)
    
    draw_multi_lines(Adens_np_day,'Anp',plotAnp_folder,a,w,Tmin = TminA,Tmax = TmaxA,xlabel = 'Day', ylabel = 'Aphid Population Abundance')
    draw_multi_lines(Adens_p_day,'Ap',plotAp_folder,a,w,Tmin = TminA,Tmax = TmaxA,xlabel = 'Day', ylabel = 'Aphid Population Abundance')
    draw_multi_lines(Ldens_p_day,'L',plotL_folder,a,w,Tmin = TminL,Tmax = TmaxL,xlabel = 'Day', ylabel = 'Ladybird Population Abundance')        
    draw_multi_scatters(Adens_p_day,Ldens_p_day,plotAL_folder,a,w,Tmin = TminA,Tmax = TmaxA, xlabel = 'Aphid Population Abundance', ylabel = 'Ladybird Population Abundance')

    toc = time.time()
    print(temp_fn + " " + "Elapsed time: {}".format(toc - tic))
    return True

def batch20_80(i,export_fns_2080):
    df = pd.read_csv('../outputs/parameter_changes/a_w_changes_2020_2080.csv')
    df = df[args.start_idx:args.end_idx]
    a = np.array(df.a)
    w = np.array(df.w)
    a_20 = a[i]
    w_20 = w[i]
    if alter == 'aw00':
        a_change_ = 0
        w_change_ = 0
    elif alter == 'aw04':
        a_change_ = 0
        w_change_ = 4
    elif alter == 'aw08':
        a_change_ = 0
        w_change_ = 8
    elif alter == 'aw40':
        a_change_ = 4
        w_change_ = 0
    elif alter == 'aw-40':
        a_change_ = -4
        w_change_ = 0
    elif alter == 'aw44':
        a_change_ = 4
        w_change_ = 4
    elif alter == 'aw48':
        a_change_ = 4
        w_change_ = 8
    elif alter == 'aw-44':
        a_change_ = -4
        w_change_ = 4
    elif alter == 'aw-48':
        a_change_ = -4
        w_change_ = 8
    a_80 = a_20 + a_change_
    w_80 = w_20 + w_change_
    batch(a_80,w_80,export_fns_2080, year= '2080')

def folders(year):
    fold_dir = f"../outputs/exports_case{case}_{alter}"
    export_fns = []
    for folder_name in ["data", "plotAnp", "plotAp", "plotL", "plotAL"]:
        folder = os.path.join(
            fold_dir, "{}_{}".format(args.scenario, folder_name))
        assert os.path.exists(folder), "{} doesn't exist!".format(folder)
        export_fns += [folder]
    return export_fns

if __name__ == '__main__':
    export_fns_2080 = folders('2080')
    if case == 0:
        TminA = 10
        TmaxA = 35
        TminL = 10
        TmaxL= 35 
    elif case == 1:
        TminA = 10
        TmaxA = 35
        TminL = 6
        TmaxL = 35
    elif case == 2:
        TminA = 10
        TmaxA = 35
        TminL = 14
        TmaxL = 35
    elif case == 3:
        TminA = 10
        TmaxA = 35
        TminL = 10
        TmaxL = 31
    elif case == 4:
        TminA = 10
        TmaxA = 35
        TminL = 10
        TmaxL = 39
    elif case == 5:
        TminA = 10
        TmaxA = 35
        TminL = 6
        TmaxL = 31
    elif case == 6:
        TminA = 10
        TmaxA = 35
        TminL = 6
        TmaxL = 39
    elif case == 7:
        TminA = 10
        TmaxA = 35
        TminL = 14
        TmaxL = 31
    elif case == 8:
        TminA = 10
        TmaxA = 35
        TminL = 14
        TmaxL = 39

    num_idxs = 1160
    for idx in np.arange(num_idxs) :
        #batch20_80(idx,export_fns_2080)
        processed_list = Parallel(n_jobs=num_cores)(delayed(batch20_80)(idx, export_fns_2080) for idx in range(num_idxs))
