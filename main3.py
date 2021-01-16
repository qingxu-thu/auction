import numpy as np
import math
import xlrd
from matplotlib import pyplot as plt

#q;v;mu;lda

g = 200
a_1 = 10
b = 4
a_2 = 0.1

def N_f(q,v):
    #return g*q-a_3/3*q*q*q-a_1*v*v*q+b*v*q+0.5*q*q*v
    return g*q-a_1/3*q*q*q+0.5*b*v*q*q-a_2*v*v*q

def N_v(q,v):
    return 0.5*b*q*q-2*a_2*v*q

def inverse_NI(mu,v,rho):
    #delta = v*v+4*a_3*(g-a*v*v+b*v+2/rho*a*v-1/rho*b-mu)
    delta = (b*v-a_2*v*v-b/rho)*(b*v-a_2*v*v-b/rho)-4*(-a_1*(g-mu+2*a_2*v/rho))
    if delta>0:
        x = (-(b*v-a_2*v*v-b/rho)-math.sqrt(delta))/(2*(-a_1))
        #print(x)
        if x<0:
            #print("no solution delta")
            return -1000
        else:
            #print(x)
            return x
    else:
        #print("no solution")
        return -1000

def I_cal(q,v,alpha,lda,rho):
    if alpha<lda:
        I = N_f(q,v)-N_v(q,v)/rho-alpha*q
    else:
        I = N_f(q,v)-N_v(q,v)/rho
        #print("I",I)
    return I

def I_q0(v,rho):
    return g + 2*a_2*v/rho

def mu_max(v_list,rho_list):
    x = []
    for i in range(v_list.shape[0]):
        x.append(I_q0(v_list[i],rho_list[i]))
    return max(x)

def inverse_I(mu,alpha,lda,v,rho):
    if alpha<lda:
        return inverse_NI(mu+alpha,v,rho)
    else:
        return inverse_NI(mu,v,rho)

def allocation(N,lda,mu_list,alpha_list,v_list,B_list,q_0,rho_list):
    demand_acc = 0
    for itr,mu in enumerate(mu_list):
        #print(mu)
        q = []
        q_acc = 0
        for i in range(N):
            k = inverse_I(mu,alpha_list[i],lda,v_list[i],rho_list[i])
            if k==-1000:
                x = 0
            else:
                x = k
            q.append(x)
            q_acc += q[-1]
        #print(q_acc,q_0)
        if q_acc<q_0:
            #print("mu",mu)
            break
    for i in range(N):
        if alpha_list[i]<=lda:
            demand_acc += q[i]+B_list[i]
    #print(demand_acc)
    return demand_acc, q, mu, q_acc


def demand_satisfy(N,lda_list,mu_list,alpha_list,v_list,B_list,q_0,d,rho_list,vmax,vmin):
    for lda in lda_list:
        #print(lda)
        demand, q, mu, q_acc = allocation(N,lda,mu_list,alpha_list,v_list,B_list,q_0,rho_list)
        #print(demand,d)
        if demand > d:
            R = revenue_cal(N,q,alpha_list,v_list,lda,rho_list)
            return q,lda,R

def demand_sample(N,alpha_list,mu_list,B_list,q_0,d,vmax,vmin,epoch):
    lda_list = []
    R_list = []
    q_list = []
    mu_list_use = []
    q_acc_list = []
    for itr in range(epoch):
        print("iteration",itr)
        v_list = np.random.rand((N))*(v_max-v_min)+v_min
        R = ground_truth(N,alpha_list,mu_list,B_list,q_0,d,vmax,vmin,v_list)
        interval = 0.1 * (1/((itr+1)*(itr+1)*(itr+1)))
        if itr == 0:
            v_rounding = rounding(N, alpha_list, np.zeros((1,1)), v_list.reshape((-1, 1)), interval, vmax, vmin)
        else:
            v_rounding = rounding(N, alpha_list, lda_list, v_sample, interval, vmax, vmin)
        for lda in np.sort(alpha_list):
            v_list_rounding = rounding(N, alpha_list, lda * np.ones((1,1)),v_list.reshape((-1, 1)),interval, vmax, vmin)
            v_list_rounding = v_list_rounding.reshape([-1])
            #print(np.linalg.norm((v_list_rounding-v_list),1))
            #print(v_list_rounding.shape,v_list.shape)
            #dis_sample = sample_stat(N, v_rounding, v_max, v_min, interval)
            #q = all_F_q_cal(N,dis_sample,v_list_rounding,v_min,interval)
            #v_list_after = q_v(N,v_min,v_max,q)
            v_list_after = v_q_v(N, v_rounding ,v_max, v_min, v_list_rounding ,interval)
            rho_list = rho_cal(N,v_max,v_min,v_list_after)
            demand, q, mu, q_acc = allocation(N, lda, mu_list, alpha_list,v_list_after,B_list,q_0,rho_list)
            if demand>d:
                R_list.append(abs(revenue_cal(N,q,alpha_list,v_list_after,lda,rho_list)-R))
                #print("R_list",R_list[-1],"R",R)
                lda_list.append(lda)
                q_list.append(q)
                mu_list_use.append(mu)
                q_acc_list.append(q_acc)
                break 
        if itr == 0:
            v_sample = v_list.reshape((-1, 1))
        else:
            v_sample = np.concatenate((v_sample,v_list.reshape((-1,1))),1)
    return R_list,lda_list,q_list,mu_list_use,q_acc_list

def v_q_v(N, v_rounding ,v_max, v_min, v , interval):
    for i in range(N):     
        mask = np.array(np.unique(v_rounding[i]))
        mask = np.sort(mask)
        tmp = 0
        for j in range(mask.shape[0]):
            if v[i]>mask[j]:
                a = v[i]
                tmp += np.sum(v_rounding[i]==mask[j])
                #print(tmp)
            else:
                q = tmp/v_rounding.shape[1]
                if q==1:
                    v[i] = v_max[i]
                elif q==0:
                    v[i] = v_min[i]
                else:
                    tmp2 = (tmp+np.sum(v_rounding[i]==mask[j]))/v_rounding.shape[1]
                    q = np.random.uniform(1)*(tmp2-tmp)+tmp
                    v[i] = (v_max[i]-v_min[i])*q+v_min[i]
                    #print("rounding",v[i]-a)
    return v



def revenue_cal(N,q,alpha_list,v_list,lda,rho_list):
    I = 0
    for i in range(N):
        I+=I_cal(q[i],v_list[i],alpha_list[i],lda,rho_list[i]) 
    return I

def rounding(N,alpha_list,lda_list,v_list,interval,vmax,vmin):
    v = np.zeros(v_list.shape)
    for i in range(N):
        for j in range(v_list.shape[1]):
            if alpha_list[i]<lda_list[j]:
                v[i,j] = (int((v_list[i,j]-vmin[i])/interval)+1)*interval+v_min[i]
            else:
                v[i,j] = v_max[i]-(int((v_max[i]-v_list[i,j])/interval)+1)*interval
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


def sample_stat(N,F_sample,v_max,v_min,interval):
    dis_array = []
    for i in range(N):
        fre, point = np.histogram(F_sample, bins=int((v_max[i]-v_min[i])/interval)+1, density=True)
        F = np.cumsum(fre)
        dis_array.append(F)
    return dis_array


def all_F_q_cal(N,dis_array,v_list,v_min,interval):
    q = np.zeros((N))
    for i in range(N):
        q[i] = q_quantile_cal(dis_array[i],v_list[i],v_min[i],interval)
    return q

def q_v(N,v_min,v_max,q):
    v = np.zeros((N))
    for i in range(N):
        v[i] = (v_max[i]-v_min[i])*q[i]+v_min[i]
    return v

def q_quantile_cal(F_dis,v,v_min,interval):
    a = int((v-v_min)/interval)
    if a+1>= F_dis.shape[0]:
        q = 1
    else:
        q = np.random.uniform(F_dis[a],F_dis[a+1],1)
    return q

def draw(R_list,R):
   num = R_list.shape[0]
   #R_array = np.array(R_list)
   R_2 = R*np.ones((num))
   plt.plot(R_list)
   plt.plot(R_2)
   plt.legend()
   plt.xlabel("sample number")
   plt.ylabel("Revenue")
   plt.show()

def ground_truth(N,alpha_list,mu_list,B_list,q_0,d,v_max,v_min,v_list):
    #print(v_list)
    rho_list = rho_cal(N,v_max,v_min,v_list)
    mu_m = mu_max(v_list,rho_list)
    mu_list = np.linspace(0,mu_m,1000) 
    #print(mu_list)
    q_0 = prop*np.sum(B_list)
    d = (np.sum(B_list)+q_0)*prop_2
    q,lda,R = demand_satisfy(N,lda_list,mu_list,alpha_list,v_list,B_list,q_0,d,rho_list,v_max,v_min)
    return R



if __name__ == "__main__":
    name = './data.xlsx'
    prop = 0.02
    prop_2 = 0.35
    B_list,alpha_list =  read(name)
    lda_list = np.sort(alpha_list)
    #print(lda_list)
    N = B_list.shape[0]
    v_max = np.maximum(np.random.rand((N))*10,50)
    v_min = np.maximum(v_max-np.random.rand((N))*2,4)
    v_list = np.random.rand((N))*(v_max-v_min)+v_min
    print(v_list)
    rho_list = rho_cal(N,v_max,v_min,v_list)
    mu_m = mu_max(v_list,rho_list)
    mu_list = np.linspace(0,mu_m,1000) 
    q_0 = prop*np.sum(B_list)
    d = (np.sum(B_list)+q_0)*prop_2
    q,lda,R = demand_satisfy(N,lda_list,mu_list,alpha_list,v_list,B_list,q_0,d,rho_list,v_max,v_min)
    epoch = 200
    for it in range(50):
        R_list,lda_list,q_list,mu_list_use,q_acc_list = demand_sample(N,alpha_list,mu_list,B_list,q_0,d,v_max,v_min,epoch)
        rarray = np.array(R_list)
        if it ==0:
            finalR = rarray
        else:
            finalR = finalR + rarray
    draw(finalR/50,0)
