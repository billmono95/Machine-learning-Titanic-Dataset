# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:39:02 2020

@author: Bill
"""
import classifier 
import matplotlib.pyplot as plt

from classifier import LogisticRegression 

import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")



X_train = df.drop('survived',axis = 1)
#test to see what are the feature that are really important for the learning of the model
delete = ['survived','age','sibling','parents','ticket','fare']
#my titanic experience 
my_titanic = [1, 0, 24.0,0, 0,7.000]
died = ['sex','survived']
Positive_train = df.drop(delete, axis  = 1)
Negative_train = df.drop(died, axis = 1)



y_train = df['survived']


X_test = df2.drop('survived',axis = 1)
y_test = df2['survived']


        
prova = LogisticRegression()
'''
prova2 = LogisticRegression()

prova3 = LogisticRegression()
'''

w,b = prova.fit(X_train,y_train)
#fit the model only with sex feature
'''
wp,bp = prova2.fit(Positive_train,y_train)

#fit the model without sex feature
wn,bn = prova3.fit(Negative_train,y_train)
'''

#parameters
print(f' Bias {prova.b} and Weight {prova.w}')
pred = np.round(prova.predict(X_train,w,b))
# MY PROBABILITY OF SURVIVED
chance = (prova.predict(my_titanic,w,b))
print(f'My possibility to survive {chance*100}')


# ACCURACY OF THE TRAINING
accuracy_train = (pred == y_train).mean()
print(f'Accuracy of the train prova {accuracy_train*100}')


#ACCURACY OF THE TEST
pred_test = np.round(prova.predict(X_test,w,b))
accuracy_test = (pred_test == y_test).mean()
print(f'Accuracy of the test prova {accuracy_test *100}')

'''
#ACCURACY OF THE TEST ONLY WITH SEX FEATURE
pred_test1 = np.round(prova2.predict(Positive_train,wp,bp))
accuracy_test1 = (pred_test1 == y_train).mean()
print(f'Accuracy of the test prova sex feature {accuracy_test1 *100}')

#ACCURACY OF THE TEST WITHOUT SEX FEATURE
pred_test2 = np.round(prova3.predict(Negative_train,wn,bn))
accuracy_test2 = (pred_test2 == y_train).mean()
print(f'Accuracy of the test prova without sex feature {accuracy_test2 *100}')
'''
#PROBABILITY TO SURVIVE

df3=df[df['survived'] == 1]

#sex feature
fig = plt.figure()
plt.hist(df3['sex'], color = 'r')

plt.title('Survivors sex features')

fig.savefig('plot.png')

plt.show()

#ticket feature survivors
df4=df[df['survived'] == 0]
fig4 = plt.figure()

plt.hist(df4['ticket'], color = 'm')

plt.title('Died Ticket features')

fig4.savefig('plot4.png')

plt.show()


#age feature survivors
fig2 = plt.figure()
plt.hist(df3['age'], color = 'b')

plt.title('Survivors age features')

fig2.savefig('plot2.png')

plt.show()



#age feature died
df4=df[df['survived'] == 0]
fig3 = plt.figure()
plt.hist(df4['age'], color = 'c')

plt.title('Died age features')

fig3.savefig('plot3.png')

plt.show()



#scatter 

fig5 = plt.figure()
plt.scatter(df['sex'],df['ticket'], c = df['survived'] )
plt.xlabel("sex")
plt.ylabel("ticket")
plt.title('Scatter between the two most influential features.')

fig5.savefig('plot5.png')

plt.show()



