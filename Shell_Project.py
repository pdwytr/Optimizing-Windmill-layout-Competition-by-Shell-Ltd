#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import random


# In[2]:


from AEP_calculator import get_AEP,checkConstraints


# In[3]:


from chromosome import fn_chromosome,fn_plot,fn_generate_initial_population


# # Genetic Algorithm for Turbine Points Layout
# 1. Gene for this problem would be x,y co-ordinates of one turbine.
# 2. Chromosome would be 50 x,y co-ordinates for 50 turbines(farmLayout design).
# 3. There are two constraints in this problem.They are Proximity and perimeter constraints. 
# 4. Total Aep Value would be calculated based on each chromosome and elite population is found.
# 5. Crossover Strategy: it takes top 4 layouts provided by sample elite population function as input and randomly picks some 50 points out of them and creates one child layout and adds it to the sample elite population.so that if the child is better than the parents then it will be picked as top layout if not parents will be picked.
# 6. Mutation Strategy:it takes top 4 layout provided by sample elite population function as input and selects one random layout out of the 4 layouts. from that one layout it retains 30 points and randomly generates 20 new points and adds to it to create one mutated child of 50 points and adds this child to the top 4 layouts.so if the mutated child is better than the parent it will be picked up as top layout if not our sample elite population will be retained in the next iteration.
# 

# # Optimization Function
# 
# 

# In[4]:


turb_diam = 100


# #Constraints

# # Evolution

# ## 1.Crossover

# In[5]:


#randomly choose 50 coordinates from top 4 layouts and check if they are satisfying constraints 
def fn_crossover(layout_list):
    layout = layout_list.copy()
    flattened_df = pd.DataFrame(columns = ['x','y'])
    flattened_df =pd.concat(layout_list)
    flattened_df = flattened_df.sample(frac = 1) 
    flattened_df.reset_index(drop = True, inplace =True)
    Xover_coords = flattened_df.to_numpy()
    candidate_index = np.random.randint(low = 0,  high=Xover_coords.shape[0], size=1)
    df = pd.DataFrame(index = [0],columns = ['x','y'])
    df.loc[0] = Xover_coords[candidate_index]
    Xover_coords = np.delete(Xover_coords, candidate_index, axis = 0)
    
                # checks if for every turbine perimeter constraint is satisfied. 
                # breaks out if False anywhere
                # for turb in mutant_coords:
                # turb = Point(turb)
                # inside_farm   = farm_poly.contains(turb)
                # correct_clrnc = farm_poly.boundary.distance(turb) >= bound_clrnc
                # if (inside_farm == False or correct_clrnc == False):
                    #peri_constr_viol = True
                    #break
                # checks if for every turbines proximity constraint is satisfied. 
                # breaks out if False anywhere
                
          
    for i,turb1 in enumerate(Xover_coords):
        
        flag = 0
        #for turb2 in np.delete(df, i, axis=0):
        for j in range(df.shape[0]):
            turb2 = np.array(df.iloc[j])
            #turb2 = np.array(turb2)
            #print(type(turb1))
            #print(turb2)
            if  np.linalg.norm(turb1 - turb2) < 4*turb_diam:
                flag = 1
                break
                
        if flag ==0: 
            
            #df.loc[len(df),'x'] = turb1[0]
            #df.loc[(len(df)-1),'y'] = turb1[1]
            
            df.loc[len(df)] = turb1
            if (len(df)==50):
                break
            
                #crossover_list.append(turb1)
                
                #    np.unique(a, axis=0)


        
        
    #return df
    #layout.append(df)
    return df


# ## 2.Choose Elite

# In[6]:


def fn_sample_elite(new_population,iteration_no):
    AEP_df = pd.DataFrame(columns = ['AEP','iteration_num'])
    
    for i in range (len(new_population)):
        AEP_df.loc[len(AEP_df),'iteration_num'] = iteration_no
        AEP_df.loc[(len(AEP_df)-1),'AEP'] = get_AEP(new_population[i])

     
    elite = AEP_df['AEP'].sort_values(ascending = False).to_frame()[0:4]
    print(elite.index)
    global elite2
    elite2 = AEP_df.sort_values(by = ['AEP'],ascending = False)
    print(elite2)
    
    var = list(elite.index.values)
    
    #print(var)
    #print(elite.index.values)
    elite_layout = [new_population[i] for i in var]
    print('sample elite output type is:',type(elite_layout))
    
    return elite_layout


# In[7]:


mutation_probability =0.8
number_of_chromos_in_population =15
number_of_iterations = 25


# ## 3.Evolute

# In[8]:


def fn_evolution(elite_population_layouts,
                  mutation_probability,number_of_chromos_in_population):
    new_pop = pd.DataFrame(index = [0],columns=['x','y'])
    #print(new_pop)
    elite_layout = elite_population_layouts.copy()
    print("elite_layout type is:",type(elite_layout))
    new_pop_iterations = 0
    while (new_pop_iterations < number_of_chromos_in_population):
        #print( iteration)
        #print("fn_evaluation new_pop_iterations", new_pop_iterations)
        #new_pop_iterations += 1 
        if np.random.random() < mutation_probability:
            print("mutation started")
            #print("fn_evaluation if")
            # The candidate (or one of the elite portfolios) is chosen randomly for mutation.
            candidate_id = np.random.randint(low = 0,
                                             high=len(elite_population_layouts))
            
            
            new_pop = fn_chromosome()
            print("lenght of elite layout in mutation before appending is :", len(elite_layout))
            elite_layout.append(new_pop)
            print("lenght of elite layout in mutation after appending is :", len(elite_layout))
            
            
            # We prefer to explore much more in the beginning of the search process to 
            # ensure diversity and avoid local optimum. 
            # As we progress towards the end of search process ( or iterations), 
            # we need to ensure the convergence of the population to a good solution 
            # if not the best. Hence we keep on reducing mutation probability with 
            # each iteraation
            
            mutation_probability = mutation_probability/(new_pop_iterations + 1)
            print("mutation process is taking progress")
            new_pop_iterations = new_pop_iterations+1
        else:
              new_pop = fn_crossover(elite_population_layouts)
              print('type of crossover output is :',type(new_pop) )
        
                
              new_pop_iterations = new_pop_iterations+1
              print("lenght of elite layout in cross over before appending is :", len(elite_layout))
              elite_layout.append(new_pop)
              print("lenght of elite layout in cross over after appending is :", len(elite_layout))
            
    print('type of evaluation output is :',type(elite_layout) )
    return elite_layout


# # Initiation of main

# In[9]:


def intitiate():

    layout1 = pd.read_csv("layouts/s1.csv")
    layout2 = pd.read_csv("layouts/s2.csv")
    layout3 = pd.read_csv("layouts/s3.csv")
    layout4 = fn_chromosome()

    return [layout1,layout2,layout3,layout4]


# In[15]:


if __name__=="__main__":
    list_layout = fn_generate_initial_population(4)
    
    new_population = list_layout.copy()
    

    print("type of new population is :",type(new_population))
    
    
    # Run genetic algorithm for number_of_iterations times.
    AEP_df = pd.DataFrame(columns = ['AEP','iteration_num'])
    for i in range(number_of_iterations):
        elite_population = fn_sample_elite(new_population,i)                                               
        
        print("iteration:",i)
        print("length of new population in main function is :",len(elite_population))
        new_population = fn_evolution(elite_population,mutation_probability,number_of_chromos_in_population)

        
        
        

        print_counter = i % 100
        if print_counter == 0:
            print("iteration", i)

    AEP_df = pd.DataFrame(columns = ['AEP'])
    for k in range (len(new_population)):
        AEP_df.loc[len(AEP_df),'AEP'] = fn_fitness_function(new_population[k])
     
    elite = AEP_df['AEP'].sort_values(ascending = False).to_frame()[0:1]
    print(elite)
    print(elite.index.values)
    new_population = new_population[elite.index.values[0]]
    

    print(new_population)


# In[11]:


k = fn_main()


# In[ ]:


fn_plot(a)


# In[ ]:


a = pd.read_csv('submissionk4.csv')


# # Submission

# In[ ]:


k.to_csv("submission.csv",index=False)


# In[ ]:




