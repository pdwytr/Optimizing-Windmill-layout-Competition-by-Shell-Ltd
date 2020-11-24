# Shell.ai Hackathon for Sustainable and Affordable Energy

The world needs to move to a cleaner energy system if it is to meet growing energy demand while tackling climate change. In April 2020, Shell shared its ambition to become a net-zero emissions energy business by 2050, or sooner.

Renewable electricity is central to this ambition. Electricity is the fastest-growing part of the energy system and, when generated from renewable sources such as wind, has a big role to play in reducing greenhouse gas emissions. We see digitalisation and AI as key enablers to the energy transition.

## Challenge: Windfarm Layout Optimisation

In this Shell.ai Hackathon for Sustainable and Affordable Energy, we invite you to optimise the placement of 50 wind turbines of 100 m height and 100 m rotor diameter each on a hypothetical 2D offshore wind farm area such that the AEP (Annual Energy Production) of the farm is maximized. One of the key problems of an unoptimized layout is the combined effect wind turbines can have on the wind speed distribution in a windfarm. As a wind turbine extracts energy from incoming wind, it creates a region behind it downstream where the wind speed is decreased- this is called a wake region. Note that wind turbines automatically orient their rotors, to face incoming wind from any direction. Due to the induced speed deficit, a turbine placed inside the wake region of an upstream turbine will naturally generate reduced electrical power. This inter-turbine interference is known as a wake effect. An optimal windfarm layout is important to ensure a minimum loss of power during this combined wake effect.

This Shell.ai Hackathon for Sustainable and Affordable Energy edition, focuses on an interesting and complex coding problem. When competing, you will face challenges such as a high dimensionality, complex multimodality and the discontinuous nature of the search space. This makes optimising the layout analytics difficult. But, armed with optimization strategies and computer algorithms, you can solve this problem.

## 1. Introduction
The energy transition and digitalisation are two mega-trends that will affect the world in the coming decades. A planet with more people and rising living standards will need more and cleaner energy solutions. We must reduce carbon emissions to tackle climate change. Shell believes that digitalisation and AI are critical enablers to support our ambition to be a net zero energy company. We see great potential in producing renewable power from wind. This hackathon, powered by Shell.ai and, provides an opportunity to work on a challenge commonly met with wind farm layout designs: how to find the most optimal and profitable layout of turbines inside a wind farm. Optimal layout of wind turbines in a wind farm carries huge business importance. An unoptimized or suboptimal layout can typically cost upto 5-15 % of AEP (Annual Energy Production) [1], which eliminates the business case. Simultaneously, it also has the potential to steer the energy portfolio further towards sustainable and cleaner energy. The key problem with an unoptimized layout is the combined influence of arranged wind turbines on the wind speed distribution across the limited area of farm. As a wind turbine extracts energy from the incoming wind, it creates a region behind it downstream where the wind speed is decreased – wake region. Note that wind turbines automatically orient their rotors to face the incoming wind from any direction. Due to the induced speed deficit, a turbine placed inside the wake region of an upstream turbine will naturally generate reduced electrical power. This inter-turbine interference is known as wake effect. In order to deal with it, strategies to layout a wind farm in an optimal fashion should be employed such that the power losses incurred due to the combined wake effect is minimum. Optimizing the layout of a wind farm is an interesting and complex optimization problem. The key challenge arises due to the high dimensionality, complex multimodality and discontinuous nature of the search space. Hence, optimizing the layout analytically is impossible. A smarter approach towards solving this problem is through optimization strategies and computer algorithms.

## 2. Problem Statement
Before we present the problem statement, we would like to mention clearly that in practice wind farm layout optimization is a complex problem, and in here we are only presenting a simpler version to test numerical optimization capabilities. We specify the assumptions and simplifications at suitable places in this text. In this hackathon, the challenge is to optimize the placement of Nturb, 50 wind turbines of 100 m rotor diameter and 100 m height each on a hypothetical 2D offshore wind farm area such that the AEP (Annual Energy Production) of the farm is maximized. The farm area for this problem is square in shape having dimensions: length Lx = 4 km, width Ly = 4 km. The orientation of farm is such that one of its edges is parallel to the geographical North direction. See Figure 1. There it can be assumed that the positive y and x axis shown are aligned with the geographical North and East directions respectively. There are two constraints that must not be violated by a wind farm layout to be considered valid.
- Perimeter Constraint. All the turbines must be located inside the perimeter of the farm, while maintaining a minimum clearance of 50 meters from the farm boundary.
- Proximity Constraint. The distance between any two turbines must be larger than a given security threshold to ensure a longer lifetime of the turbine rotors. This minimum distance between two turbines is called Dmin and needs to be 400 m. 

Further details pertaining to problem statement maybe found in the attached [problem-statement pdf](https://github.com/pdwytr/Optimizing-Windmill-layout-Competition-by-Shell-Ltd./blob/main/problem-statement.pdf)

## 3. Solution Summary
### 1. Approach

Genetic Algorithms approach is used to  maximize the totalAEP of the layout

- Chromosome: Randomly generated 50 points layout satisfying perimeter and proximity constraints
- Fitness Function: Function to improve TotalAEP of the farm
- New population generation : Crossover of top layouts or Mutation
- Selection: it is based on Highest Total Aep layouts from elite population  

### 2. Algorithm Engineering

- initial population: previous top performing layouts or random layouts if not available
- Crossover: Select 50 points from randomly from the parent layouts
- Mutation: Select portion of edge points plus randomly selecting remaining of the 50 points satisfying constraints
- Velocity-Direction Profile: Weighted averages previously available 3 years of Velocity-Direction data

### 3. Tools used

- Google colabs
- Jupyter notebooks

# Results:
Finished at 32 postion in top 10 percent of non-zero-scoring competitiors as [Shaik-Mohammed-Khalid-Naveed-Team](https://www.hackerearth.com/challenges/competitive/shell-hackathon/leaderboard/)

