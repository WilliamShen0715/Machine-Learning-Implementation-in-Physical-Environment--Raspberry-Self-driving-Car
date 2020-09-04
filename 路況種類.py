# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:55:28 2020

@author: William
"""

count=0;
a=b=c=0;
print("路況種類:")
for a in range(0,4):
    for b in range(0,4):
        for c in range(0,4):
            if (a==0 and b==0 and c==0) or (a+c<=3 and (a!=0 or c!=0)):
                #count = count+1
                if b<2:
                    count = count+1
                    print(count-1,": ",a,b,c)