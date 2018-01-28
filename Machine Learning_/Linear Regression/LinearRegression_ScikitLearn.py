# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:33:03 2018

@author: hazel.turan
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\hazel.turan\\Desktop\\Jupyter File\\Machine Learning\\linear.csv")


x = data["metrekare"] #metrekareler bir axis'e çekilir pandas'ın özelliği
y = data["fiyat"] #fiyatlar bir axis'e çekilir pandas'ın özelliği

x= x.reshape(100,1) #99*1 lik matris
y= y.reshape(100,1)

linearregression=lr()
linearregression.fit(x,y) #x ve y değerlerini grafiğe oturttuk ve tahmini bi çizgi çektik.
linearregression.predict(x) #predict tahmin ediyoruz x eksenine göre

m=linearregression.coef_
b=linearregression.intercept_

#mx+b
print("Eğim:" , linearregression.coef_) #coef_ katsayı demek
print("Y ekseninde kesiştiği nokta:",linearregression.intercept_) #intercept_ kesişim demek

print("Denklem:")
print("y=",m,"x+",b)

a=np.arange(150)

plt.scatter(x,y)
plt.plot(a,m*a+b)
plt.show()

