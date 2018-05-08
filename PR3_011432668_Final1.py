
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import math
from scipy.spatial import distance
from sklearn import preprocessing


# In[2]:


#converting the given data into a CSR matrix -- code given
def build_CSR_Matrix(file_name):
    nidx=1
    with open(file_name) as fi:
        allLines = fi.readlines()
    #remove whitespace characters and`\n` from end
    allLines = [x.strip() for x in allLines] 
    
    nrows = len(allLines)
    ncols = 0 
    nnz = 0 
    for i in xrange(nrows):
        p = allLines[i].split()
        nnz += len(p)/2
        for j in xrange(0, len(p), 2): 
            cid = int(p[j]) - nidx
            if cid+1 > ncols:
                ncols = cid+1
    val = np.zeros(nnz, dtype=np.float)
    ind = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.long)
    n = 0 
    for i in xrange(nrows):
        p = allLines[i].split()
        for j in xrange(0, len(p), 2): 
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j+1])
            n += 1
        ptr[i+1] = n 
    return csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    


# In[3]:


#receiving the data from the csr matrix into Data
Data=build_CSR_Matrix('train.dat')


# In[4]:


#normalizing the data using sklearn l2 normalizer

Data = preprocessing.normalize(Data, norm='l2')


# In[5]:


#performing a dimensionality reduction using SVD sklearn library
#used components = 50 ; NMI was 0.38
#using components = 50 NMI 0.07 
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
Data= svd.fit_transform(Data)


# In[ ]:


#initializing a global matrix
#The shape is 8580*8580
matrixDistance= np.zeros((8580,8580)) #8580*8580


# In[6]:


#Calculating the distance matirx - distanceof every point with every other point. 

def findDistanceMatrix(matrixDistance, row_current, row, mat):
    row=row+1 #finding distances from the next row, leaving itself out
    for col in range(row, 8580):
        #calculate distance -- Euclidean distance and assigning it to the respective index.
        dist = math.sqrt(np.power(row_current-mat[col,:],2).sum())
        matrixDistance[row][col] = dist
        matrixDistance[col][row] = dist


# In[7]:


for row in range(0, 8580):
    row_current =  Data[row, :]
    findDistanceMatrix( matrixDistance,row_current, row, Data)


# In[8]:


print matrixDistance


# In[9]:


print matrixDistance.shape[0]


# In[9]:


#declaring global variables to be accessible by the printing out statment
cluster_pts=[]
noise_pts = []
core_pts = []
border_pts=[]
neighbors=[]
new2=[]
visited=[]

def checkCLuster(index):
    for clust in cluster_pts:
        for k in range(len(clust)):
            if clust[k]==index:
                pass_index=True
                return pass_index

            #main dbscan function
def DBSCAN(D, density, eps):
    
    dense = 0
    index_row= cindex = -1
    
    global noise_pts
    global cluster_pts
    pass_index = False
    
    #loop over each row in the distance matrix
    for points in matrixDistance:
        #row in dist_matrix
        index_row+=1
        
        #check if the row has been visited. 
        #If yes, then move to the next point. else continue executing other statements
        if index_row in visited: continue
           
        #loop over every attribute(here distance) in each row 
        for i in range(len(points)):
            if points[i] <= eps and points[i]!=0:
                dense += 1
                neighbors.append(i)
        
        #add the index of the current row to the visited_points list
        visited.append(index_row)
        
        #check if the number of points in the neighbor list is < density -- 
        #If yes, then classify the point as a NOISE point
        if dense < density:
            noise_pts.append(index_row)
            dense=0
        
        else:
            #append the index of the current row to a cluster, and to the set of CORE POINTS
            cindex+=1
            #appendtoCore(index_row)
            core_pts.append(index_row)
            #creating a new cluster of cluster and appending first index in the cluster index = 'cindex'
            cluster_pts.append([])
            cluster_pts[cindex].append(index_row)
            
            
            #loop over the neighbor_points 
            for neighbor_pts in neighbors:
                
                #check if the point has been visited or not
                if neighbor_pts not in visited:
                    
                    #if not. append to the visited list
                    visited.append(neighbor_pts)
                    
                    #check for thepoints' neighbors in the distance matrix for that point
                    matrix_part = matrixDistance[neighbor_pts,:]
                    for j in range(len(matrix_part)):
                        #append to new neighbor list and check the epsilong case
                        if(matrix_part[j] <= eps):
                            new2.append(j)
                    
                    #check if the point is present in the cluster already
                    ind = checkCLuster(neighbor_pts)
                    
                    if (len(new2) >= density):
                        #to append only if not in cluster
                        core_pts.append(neighbor_pts)
                        #if no
                        if ind ==False:
                            cluster_pts[cindex].append(neighbor_pts)
                        else:
                            ind = False
                    else:
                        if ind ==False:
                            border_pts.append(neighbor_pts)
                            cluster_pts[cindex].append(neighbor_pts)
                        else:
                            ind = False
                       
            dense=0  
    
    print len(cluster_pts)
    print len(noise_pts)
    print len(core_pts)
    print len(border_pts)
                    
        


# In[10]:


DBSCAN(Data, 21, 0.40)


# In[ ]:


#initializing a temp list with 8580 columns
temp= [0]*8580 #8580*8580
for i in range(8580):
    for lst_index in range(0, len(cluster_pts)):
        if ind in cluster_pts[lst_index]:
            temp[ind] = lst_index+1


# In[ ]:


output_file = open("output.dat", "w")
for j in temp:
    output_file.write(str(int(j)))
    output_file.write('\n')

output_file.close()

