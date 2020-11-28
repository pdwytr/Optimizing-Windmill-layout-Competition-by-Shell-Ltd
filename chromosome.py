import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def st_layout():
    
    a=[483,916,1349,1782,2215,2648,3081,3514]
    b=[]
    for i in range(len(a)):
        
        b.append([a[i],50])
        b.append([a[i],3947])
        b.append([50,a[i]])
        b.append([3947,a[i]])
    b.append([50,3947])
    b.append([50,50])
    b.append([3947,50])
    b.append([3947,3947])
    return b



#---------------------------------------------------------------------------------------------------------------------#



def newpoint():
    while(True):
        random = np.random.rand(1,2)*4000
        if (random[0,0]>50 and random[0,0]<3950) and (random[0,1]>50 and random[0,1]<3950):
            return random.tolist()[0]
        
#---------------------------------------------------------------------------------------------------------------------#

def distance(A,B):
    return ((A[0]-B[0])**2+(A[1]-B[1])**2)**0.5

#---------------------------------------------------------------------------------------------------------------------#

def constraint_checker_for_new_point(point,layout: list) -> bool:

    for i in layout:
        if abs(distance(point,i))<=400:
            return False
    return True

#---------------------------------------------------------------------------------------------------------------------#

def fn_chromosome():
    k = st_layout()
    np.random.shuffle(k)
    a=k[:20]
    #a.append(newpoint())
    count=0
    while(len(a)!=50):
        new = newpoint()
        if constraint_checker_for_new_point(new,a):
            a.append(new)
            count+=1
    final = np.array(a).reshape(50,2)
    final = pd.DataFrame(final,columns=["x","y"])
    return final

#---------------------------------------------------------------------------------------------------------------------#

def fn_plot(s):
    plt.scatter(s['x'],s['y'])
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()

#---------------------------------------------------------------------------------------------------------------------#

def fn_generate_initial_population(population_size):
    init_population_list = []
    init_population_list = [fn_chromosome() for _ in range(population_size)]
    return init_population_list
