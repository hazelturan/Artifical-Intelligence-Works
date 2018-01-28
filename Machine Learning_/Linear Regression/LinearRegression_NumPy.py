# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:33:03 2018

@author: hazel.turan
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\hazel.turan\\Desktop\\Jupyter File\\Machine Learning\\linear.csv")
print(data)

x = data["metrekare"] #metrekareler bir axis'e çekilir pandas'ın özelliği
y = data["fiyat"] #fiyatlar bir axis'e çekilir pandas'ın özelliği

x = pd.DataFrame.as_matrix(x) #NumPy matrisine dönüşüm
y = pd.DataFrame.as_matrix(y) #NumPy matrisine dönüşüm

print(x)
print(y)

plt.scatter(x,y) # Matplotlib ile 2 boyutlu grafik üzerinde görüntüleme

#Doğru denklemi: y=m*a+b En uygun m ve b değerleri aranmakatadır.

m,b = np.polyfit(x,y,1) # np.polyfit(x ekseni,y ekseni,denklemin derecesi) Numpy ile doğru garfik üzerinde görüntülenir.

a = np.arange(150) # a nın aralığı 

plt.scatter(x,y) #nokta çizdirimi yapılır.
plt.plot(m*a+b)

z = int(input("Kaç metrekare?"))
tahmin = m*z+b
print(tahmin)

plt.scatter(z,tahmin,c="red",marker=">")

plt.show()
print("y=",m,"x+",b)