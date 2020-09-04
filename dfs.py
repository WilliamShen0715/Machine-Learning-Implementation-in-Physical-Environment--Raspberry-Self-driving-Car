# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:58:43 2020

@author: William
"""

from collections import defaultdict 
  
# This class represents a directed graph using 
# adjacency list representation 
class Graph: 
  
    # Constructor 
    def __init__(self): 
  
        # default dictionary to store graph 
        self.graph = defaultdict(list) 
  
    # function to add an edge to graph 
    def addEdge(self, u, v): 
        self.graph[u].append(v) 
  
    # A function used by DFS 
    def DFSUtil(self, v, visited): 
  
        # Mark the current node as visited  
        # and print it 
        visited[v] = True
        print(v, end = ' ') 
  
        # Recur for all the vertices  
        # adjacent to this vertex 
        for i in self.graph[v]: 
            if visited[i] == False: 
                self.DFSUtil(i, visited) 
  
    # The function to do DFS traversal. It uses 
    # recursive DFSUtil() 
    def DFS(self, v): 
  
        # Mark all the vertices as not visited 
        visited = [False] * (max(self.graph)+1) 
  
        # Call the recursive helper function  
        # to print DFS traversal 
        self.DFSUtil(v, visited)
class Gstack:
    s=[]
    route_s=[]
    route = []
    dele = []
    def __init__(self,x_init,y_init):
        self.addNode(x_init,y_init)
    def addNode(self, x, y):
        k = []
        k.append((x,y))
        self.s.append(k)
    def addEdge(self, x,y,u, v):
        for i in self.s:
            if i[0]==(x, y):
                i.append((u,v))
                self.s.append([(u,v)])
    def getRoute(self):
        b=0
        self.route_s = self.s.copy()
        self.route.append(self.route_s[0][0])
        while len(self.route_s)!=0 and b!=-1:
            b=-1
            for i in self.route_s:
#                print("i[-1]:r",i[-1],self.route[-1])
                if len(i)>1:
                    self.route.append(i[-1])
#                    self.route.append(i[-1])
                    i.pop()
                elif i[-1]== self.route[-1] and len(i)==1:
#                    print("check:")
                    back=(-1,-1)
                    for j in self.route_s:
#                        print("back len:",len(j))
                        if len(j)>1:
                            back=j[0]
                        elif j==self.route[-1]:
                            break
                    if back!=(-1,-1):
                        self.route.append(back)
                        self.route.append((-100,-100))
            for i in self.route_s:
                if len(i)>1:
                    b=i
#            for i in range(0,len(self.route_s)):
#                if len(self.route_s[i])==1:
#                    self.dele.append(i)
#            for i in self.dele:
#                print(i)
#            for i in self.dele:
#                del self.route_s[i]
        for i in self.route:
            print(i)
                    
                
# Driver code 
  
# Create a graph given  
# in the above diagram 
g = Graph() 
g.addEdge(0, 1) 
g.addEdge(0, 2) 
g.addEdge(1, 2) 
g.addEdge(2, 0) 
g.addEdge(2, 3) 
g.addEdge(3, 3) 
  
#print("Following is DFS from (starting from vertex 2)") 
#g.DFS(2)

stack = Gstack(0,1)
stack.addEdge(0,1,0,0)

stack.addEdge(0,0,1,0)

stack.addEdge(1,0,2,0)

stack.addEdge(2,0,3,0)
stack.addEdge(2,0,2,1)

stack.addEdge(2,1,2,2)

stack.addEdge(2,2,2,3)
stack.addEdge(2,2,1,2)

stack.addEdge(1,2,0,2)
stack.addEdge(1,2,4,0)
for i in stack.s:
    print(i,":")
stack.getRoute()