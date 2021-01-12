import numpy as np
import math
import xlrd
#q;v;mu;lda

g = 0.5
a_1 = 0.3
a_3 = 0.3
b = 0.3
a = 0.5

def N(q,v):
    return g*q-a_3/3*q*q*q-a_1*v*v*q+b*v*q+1/2*q*q*v

def N_v(q,v):
    return (-2*a_1*q*v+v*q)

def inverse_NI(mu,v,rho):
    delta = v*v+4*a_3*(g-a*v*v+b*v+2/rho*a*v-1/rho*b-mu)
    if delta>0:
        x = (-v+math.sqrt(delta))/(2*a_3)
        if x<0:
            print("no solution")
            return -1000
        else:
            return x
    else:
        print("no solution")
        return -1000

def I_cal(q,v,alpha,lda,rho):
    if alpha<lda:
        I = N(q,v)-N_v(q,v)/rho-alpha*q
    else:
        I = N(q,v)-N_v(q,v)/rho
    return I

def I_q0(v):
    return g - a_1*v*v + b*v

def mu_max(v_list):
    x = []
    for i in range(v_list.shape[0]):
        x.append(I_q0(v_list[i]))
    return max(x)

def inverse_I(mu,alpha,lda,v,rho):
    if alpha<lda:
        return inverse_NI(mu+alpha,v,rho)
    else:
        return inverse_NI(mu,v,rho)

def allocation(N,lda,mu_list,alpha_list,v_list,B_list,q_0,rho_list):
    for itr,mu in enumerate(mu_list):
        q = []
        q_acc = 0
        demand_acc = 0
        for i in range(N):
            if inverse_I==-1000:
                x = 0
            else:
                x = inverse_I(mu,alpha_list[i],lda,v_list[i],rho_list[i])
            q.append(x)
            q_acc += q[-1]
        if q_acc>q_0:
            break
    for i in range(N):
        if alpha_list[i]<lda:
            demand_acc = q[i]+B_list[i]
    return demand_acc, q, mu, q_acc


def demand_satisfy(N,lda_list,mu_list,alpha_list,v_list,B_list,q_0,d,rho_list,vmax,vmin,flag,interval=0):
    for lda in lda_list:
        v_list = rounding(N,alpha_list,lda,flag,v_list,interval,vmax,vmin)
        demand, q, mu, q_acc = allocation(N,lda,mu_list,alpha_list,v_list,B_list,q_0,rho_list)
        if demand>d:
            R = revenue_cal(N,q,alpha_list,v_list,lda,rho_list)
            return q,lda,R

def revenue_cal(N,q,alpha_list,v_list,lda,rho_list):
    I = 0
    for i in range(N):
        I+=I_cal(q[i],v_list[i],alpha_list[i],lda,rho_list[i]) 
    return I

def rounding(N,alpha_list,lda,flag,v_list,interval,vmax,vmin):
    if flag==0:
        return v_list
    else:
        v = [None]*N
        for i in range(N):
            if alpha_list[i]<lda:
                v[i] = (int((v_list[i]-vmin[i])/interval)+1)*interval+v_min[i]
            else:
                v[i] = v_max-(int((v_max[i]-v_list[i])/interval)+1)*interval
        return v

def rho_cal(N,vmax,vmin,v_list):
    rho = [None]*N
    for i in range(N):
        rho[i] = (1/((v_max[i]-v_min[i])))/[(v_max[i]-v_list[i])/(v_max[i]-v_min[i])]
    return rho

def read(name):
    data = xlrd.open_workbook(name)
    table = data.sheets()[0]
    B = np.array(table.col_values(1), dtype = "float64")
    alpha = np.array(table.col_values(0), dtype = "float64")
    #G = np.array(table.col_values(0), dtype = "float64")
    return B,alpha


def sample_stat(F_sample):

    

def all_F_q_cal(F_sample,v_list,v_min,interval):


def q_quantile_cal(F_dis,v,v_min,interval):
   a = int((v-v_min)/interval)
   q = np.random.uniform(F_dis[a],F_dis[a+1],1)
   return q



if __name__ == "__main__":
    name = './data.xlsx'
    prop = 0.1
    prop_2 = 0.55
    B_list,alpha_list =  read(name)
    lda_list = np.sort(alpha_list)
    N = B_list.shape[0]
    v_max = np.random.rand((N))*300
    v_min = np.maximum(v_max-np.random.rand((N))*60,15)
    v_list = np.random.rand((N))*(v_max-v_min)+v_min
    rho_list = rho_cal(N,v_max,v_min,v_list)
    mu_m = mu_max(v_list)
    mu_list = np.linspace(0,mu_m,1000) 
    q_0 = prop*np.sum(B_list)
    d = (B_list+q_0)*prop_2
    flag = 1
    q,lda,R = demand_satisfy(N,lda_list,mu_list,alpha_list,v_list,B_list,q_0,d,rho_list,v_max,v_min,flag)

    flag = 0


    interval_size = 0.01*()