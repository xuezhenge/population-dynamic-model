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
    default=24)
parser.add_argument('--K', type=int,
    default=50000000,help = "Carrying capacity")
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
alter = args.alter

#processor
num_cores = args.num_cores 
# species = args.species


def draw_multi_lines(years,x,pltt,folder,a,w,Tmin,Tmax,xlabel,ylabel):
    dt = np.arange(0,365*years,0.01)
    plt.plot(dt,x)
    plt.title(pltt + ': a = {} w = {} Tmin = {} Tmax = {}'.format(a,w,Tmin,Tmax))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fn = 'a{}_w{}_{}_{}.png'.format(a,w,Tmin,Tmax)
    out_file = os.path.join(folder, fn)
    if not os.path.isfile(out_file):
        plt.savefig(out_file,bbox_inches='tight')
    plt.close()

def draw_multi_scatters(x,y,folder,a,w,Tmin,Tmax,xlabel,ylabel):
    plt.plot(x,y)
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

def batch(a,w,TminA,TmaxA, TminL,TmaxL,export_fns):
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
    Topt_devA = TminA + (TmaxA - TminA)*(2/3)
    Topt_fecA = TminA + (TmaxA - TminA)*(2/3)
    q2_devA = get_paras_TPC(TminA,TmaxA,Topt_devA,q1_devA)
    q2_fecA = get_paras_TPC(TminA,TmaxA,Topt_fecA,q1_fecA)

    # Aphid mortality rate
    #ToptA_mid = (TminA + TmaxA)/2
    ToptA_mor1 = 15
    ToptA_mor2 = 25
    v_maxA = 0.25
    v_minA = 0.05
    a1A,a2A,b1A,b2A = get_paras_mor2(TminA,TmaxA,ToptA_mor1,ToptA_mor2,v_maxA,v_minA)

    # PARAMETER VALUES FOR LADYBIRD
    # Development
    # developmental rate
    dev_max_egg = 0.2142
    dev_max_instar1 = 0.3778
    dev_max_instar2 = 0.4101
    dev_max_instar3 = 0.2288
    dev_max_instar4 = 0.1229
    dev_max_pupa = 0.1648
    # the parameter which affect the temperature development rate
    m_devL = 1 
    q1_devL = 1.5 
    Topt_devL = TminL + (TmaxL - TminL)*(2/3)
    q2_devL = get_paras_TPC(TminL,TmaxL,Topt_devL,q1_devL)

    # coefficients for tdpr equations
    m_tdpr = 1
    q1_tdpr = 1.5
    Topt_tdpr = TminL + (TmaxL - TminL)*(2/3)
    q2_tdpr = get_paras_TPC(TminL,TmaxL,Topt_tdpr,q1_tdpr)

    # ladybird mortality rate
    #ToptL_mid = (TminL + TmaxL)/2
    ToptL_mor1 = 15 + (TminL - 10)*0.5
    ToptL_mor2 = 25 - (35 - TmaxL)/2
    #import pdb;pdb.set_trace()
    # egg
    v_min_egg = 0.02
    v_max_egg = 0.15
    a1_egg,a2_egg,b1_egg,b2_egg = get_paras_mor2(TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_egg,v_min_egg)

    # instar 1
    v_min_inst1 = 0.03
    v_max_inst1 = 0.2
    a1_inst1,a2_inst1,b1_inst1,b2_inst1 = get_paras_mor2(TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst1,v_min_inst1)

    # instar2
    v_min_inst2 = 0.02
    v_max_inst2 = 0.15
    a1_inst2,a2_inst2,b1_inst2,b2_inst2 = get_paras_mor2(TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst2,v_min_inst2)

    # instar3
    v_min_inst3 = 0.01
    v_max_inst3 = 0.1
    a1_inst3,a2_inst3,b1_inst3,b2_inst3 = get_paras_mor2(TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst3,v_min_inst3)

    # instar4
    v_min_inst4 = 0.005
    v_max_inst4 = 0.06
    a1_inst4,a2_inst4,b1_inst4,b2_inst4 = get_paras_mor2(TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst4,v_min_inst4)

    # pupa
    v_min_pupa = 0.01
    v_max_pupa = 0.08
    a1_pupa,a2_pupa,b1_pupa,b2_pupa = get_paras_mor2(TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_pupa,v_min_pupa)

    # adu
    v_min_adu = 0.015
    v_max_adu = 0.05
    a1_adu,a2_adu,b1_adu,b2_adu = get_paras_mor2(TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_adu,v_min_adu)

    # Other parameters
    K = args.K #carrying cacpacity of ladybird
    Qp = 100 # transformation rate of ladybird
    # v_max = 0.3 # maximal mortality rate for aphid and ladybird
    theta = 0.5 # ratio of female ladybird to male ladybird

 
    # def indicator(t,Tmin,Tmax):
    #     # Tmin = Tmin_fap or TminL
    #     # Tmax = TmaxL or TmaxL
    #     i = int(np.floor(t))
    #     if (t >= 5) & (np.all(temp[i-5:i+1] >= Tmin)) & (np.all(temp[i-5:i+1] <= Tmax)):
    #             return True
    #     else: return False

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
    def Solve_euler_model(var0,A_add,L_add,t_start,t_end,dt,predation):
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
        Aden = np.zeros([n_t]); Lden = np.zeros([n_t]);Aborn = np.zeros([n_t]);Lborn = np.zeros([n_t])
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
                        Lden_t, Legg_t, L1_t, L2_t, L3_t, L4_t, Lpupa_t, Lf_t, Lm_t = [L_add, 0, 0, 0, 0, 0, 0, L_add, 0]
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
                mu_inst_t = mor_rate(Temp_t, a1A, a2A, b1A, b2A, TminA,TmaxA,ToptA_mor1,ToptA_mor2,v_maxA,v_minA)
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
                #temperature dependent develoment rate fraction
                tddr_t = thorneley_france(m_devL, TminL, TmaxL, Topt_devL, q1_devL, q2_devL, Temp_t)
                #Temperature-dependent development rates for egg and pupa
                delta_egg_t = tddr_t*dev_max_egg
                delta_pupa_t = tddr_t*dev_max_pupa
                #Temperature-dependent developments rates at prey saturation
                delta_inst1_prey_saturation_t = tddr_t*dev_max_pupa
                delta_inst2_prey_saturation_t = tddr_t*dev_max_pupa
                delta_inst3_prey_saturation_t = tddr_t*dev_max_pupa
                delta_inst4_prey_saturation_t = tddr_t*dev_max_pupa
                # Mortality rate of various stages
                gamma_egg_t = mor_rate(Temp_t,a1_egg,a2_egg,b1_egg,b2_egg,TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_egg,v_min_egg)
                gamma_inst1_t = mor_rate(Temp_t,a1_inst1,a2_inst1,b1_inst1,b2_inst1,TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst1,v_min_inst1)
                gamma_inst2_t = mor_rate(Temp_t,a1_inst2,a2_inst2,b1_inst2,b2_inst2,TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst2,v_min_inst2)
                gamma_inst3_t = mor_rate(Temp_t,a1_inst3,a2_inst3,b1_inst3,b2_inst3,TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst3,v_min_inst3)
                gamma_inst4_t = mor_rate(Temp_t,a1_inst4,a2_inst4,b1_inst4,b2_inst4,TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_inst4,v_min_inst4)
                gamma_pupa_t = mor_rate(Temp_t,a1_pupa,a2_pupa,b1_pupa,b2_pupa,TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_pupa,v_min_pupa)
                gamma_f_t = mor_rate(Temp_t,a1_adu,a2_adu,b1_adu,b2_adu,TminL,TmaxL,ToptL_mor1,ToptL_mor2,v_max_adu,v_min_adu)
                gamma_m_t = gamma_f_t
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
                #if Aden[i] <1: Aden[i] = 0;Aap[i] = 0; A1[i] = 0; A2[i] = 0; A3[i] = 0; A4[i] = 0

                if Legg[i] < 0: Legg[i] = 0
                if L1[i] < 0: L1[i] = 0
                if L2[i] < 0: L2[i] = 0
                if L3[i] < 0: L3[i] = 0
                if L4[i] < 0: L4[i] = 0
                if Lpupa[i] < 0: Lpupa[i] = 0
                if Lf[i] < 0: Lf[i] = 0
                if Lm[i] < 0: Lm[i] = 0
                #if Lden[i] < 1: Lden[i] = 0;Legg[i] = 0; L1[i] = 0; L2[i] = 0; L3[i] = 0; L4[i] = 0; Lpupa[i] = 0; Lf[i] = 0; Lm[i] = 0

                # set start condition
                if Aden[i] <= Aden_t and num_change_A == 0 and j == 0:
                    Aden[i] = 0; Aap[i] = 0; A1[i] = 0; A2[i] = 0; A3[i] = 0; A4[i] = 0
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

            # The number of borned aphid and ladybird
            if k_effect_t < 0:
                Aborn[i] = 0
            else:
                Aborn[i] = f_ap_t*k_effect_t*Aap_t*dt
            Lborn[i] = f_L_t*Lf_t*dt

            # End the simulation
            # TminL + 5 temperature threshold of entering overwintering period
            if w-a < TminL + 5:
                # if Temp_t >= 10 and num_change_A == 1 and num_change_L == 1:
                #     if Aden[i] == 0:
                #         Aden = np.zeros([n_t]); Aap = np.zeros([n_t]);A1 = np.zeros([n_t]); A2 = np.zeros([n_t]); A3 = np.zeros([n_t]); A4 = np.zeros([n_t])
                #         A_end = 0; L_end = 0
                #         break
                if Temp_t < TminL and num_change_A == 1 and num_change_L == 1:
                    A_end = Aden[i];L_end = Lden[i]
                    break
                else:
                    A_end = Aden[i];L_end = Lden[i]
                if num_change_A == 1 and num_change_L == 0:
                    A_end = 0;L_end = 0

            if w-a >= TminL + 5:
                A_end = Aden[i];L_end = Lden[i]
                
        outputs = [Aden, Lden, Aborn,Lborn,A_end,L_end]
        return outputs

    #temp = temps(a,w)
    var0 = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    Temp_min = w - a
    Temp_max = w + a
    A_add = 10000000
    L_add = 50000
    years = 30
    # import pdb;pdb.set_trace()
    if Temp_max <= TminL or Temp_min >= TmaxL:
        ts = np.arange(0,365*years,0.01)
        n_t=len(ts)
        Adens_p = np.zeros([n_t])
        Ldens_p = np.zeros([n_t])
        Aborns_p = np.zeros([n_t])
        Lborns_p = np.zeros([n_t])

    elif Temp_min >= TminL + 5 and Temp_min < TmaxL:
        Aborns_p = [];Lborns_p = []
        outputs_p = Solve_euler_model(var0,A_add,L_add,t_start = 0, t_end = 365*years,dt=0.01,predation = True)
        Adens_p = outputs_p[0]; Ldens_p = outputs_p[1]
        Aborns_p = outputs_p[2];Lborns_p = outputs_p[3]
    else:
        Adens_p = [];Ldens_p = [];Aborns_p = []; Lborns_p = []
        def year_loop(A_add,L_add):
            if A_add == 0 and L_add == 0:
                Adens_p = np.zeros([36500])
                Ldens_p = np.zeros([36500])
                Aborns_p = np.zeros([36500])
                Lborns_p = np.zeros([36500])
                A_end = 0
                L_end = 0
                A_surv = 0
                L_surv = 0
                print(A_add,L_add, A_end,L_end)
            else:
                outputs_p = Solve_euler_model(var0,A_add,L_add,t_start = 0, t_end = 365,dt=0.01,predation = True)
                Aden_p = outputs_p[0]; Lden_p = outputs_p[1]
                Aborn_p = outputs_p[2];Lborn_p = outputs_p[3]
                A_end = outputs_p[4];L_end = outputs_p[5]
                Adens_p.extend(Aden_p)
                Ldens_p.extend(Lden_p)
                Aborns_p.extend(Aborn_p)
                Lborns_p.extend(Lborn_p)
                print(A_add,L_add, A_end,L_end)
            ov_surv = 1
            A_surv = A_end*ov_surv;L_surv = L_end*ov_surv
            return A_surv,L_surv
        #year 1 to year 30
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
        A_add, L_add = year_loop(A_add,L_add)
       
        Adens_p= np.array(Adens_p)
        Ldens_p= np.array(Ldens_p)
        Aborns_p = np.array(Aborns_p)
        Lborns_p = np.array(Lborns_p)
        
    AL = np.vstack([Adens_p,Ldens_p,Aborns_p,Lborns_p])
    AL = pd.DataFrame(np.transpose(AL))
    AL.columns = ['Aden_dt', 'Lden_dt', 'Aborn_dt', 'Lborn_dt']
    if not os.path.isfile(out_file):
        AL.to_csv(out_file,header=True)
    draw_multi_lines(years,Adens_p,'Ap',plotAp_folder,a,w,Tmin = TminA,Tmax = TmaxA,xlabel = 'Day', ylabel = 'Aphid Population Abundance')
    draw_multi_lines(years,Ldens_p,'L',plotL_folder,a,w,Tmin = TminL,Tmax = TmaxL,xlabel = 'Day', ylabel = 'Ladybird Population Abundance')        
    draw_multi_scatters(Adens_p,Ldens_p,plotAL_folder,a,w,Tmin = TminL,Tmax = TmaxL, xlabel = 'Aphid Population Abundance', ylabel = 'Ladybird Population Abundance')
    
    toc = time.time()
    print(temp_fn + " " + "Elapsed time: {}".format(toc - tic))
    return True

def batch20_80(i,TminA,TmaxA,TminL,TmaxL,export_fns_2080):
    df = pd.read_csv('../outputs/parameter_changes/a_w_changes_2020_2080.csv')
    df = df[args.start_idx:args.end_idx]
    a = np.array(df.a)
    w = np.array(df.w)
    a_20 = a[i]
    w_20 = w[i]
    print(a_20, w_20)
    # a_20 = 22
    # w_20 = -4
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
    batch(a_80,w_80,TminA,TmaxA,TminL,TmaxL,export_fns_2080)

def folders(year):
    fold_dir = f"../outputs/exports_case{case}_{alter}_30years"
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

    num_idxs = 73
    for idx in np.arange(num_idxs) :
        #batch20_80(idx,TminA,TmaxA,TminL,TmaxL,export_fns_2080)
        processed_list = Parallel(n_jobs=num_cores)(delayed(batch20_80)(idx, TminA,TmaxA,TminL,TmaxL,export_fns_2080) for idx in range(num_idxs))
