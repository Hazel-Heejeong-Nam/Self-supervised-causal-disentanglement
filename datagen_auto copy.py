#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd 


def projection(theta, phi, x, y, base = -0.5):
    b = y-x*math.tan(phi)
    shade = (base - b)/math.tan(phi)
    return shade
  
def gen(i,j,shade,mid):
  fig = plt.figure(figsize=(1.0,1.0), dpi=96)
  theta = i*math.pi/200.0
  phi = j*math.pi/200.0
  x = 10 + 8*math.sin(theta)
  y = 10.5 - 8*math.cos(theta)

  ball = plt.Circle((x,y), 1.5, color = 'firebrick')
  gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)

  light = projection(theta, phi, 10, 10.5, 20.5)
  sun = plt.Circle((light,20.5), 3, color = 'orange')
  shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
  ax = plt.gca()
  ax.add_artist(gun)
  ax.add_artist(ball)
  ax.add_artist(sun)
  ax.add_artist(shadow)
  ax.set_xlim((0, 20))
  ax.set_ylim((-1, 21))

  plt.axis('off')
  plt.savefig('/mnt/hazel/data/causal_data/pendulum/eval/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png')
  
if __name__ =="__main__":
  combi = [
    [-40, 105, 7, 10],
    [40, 105, 7, 10],
    [0, 60, 7, 10],
    [0, 150, 7, 10],
    [0, 105, 3, 10],
    [0, 105, 12, 10],
    [0, 105, 7, 2],
    [0, 105, 7, 19],
  ]
  
  for item in combi:
    gen(item[0],item[1],item[2],item[3])
    
    
