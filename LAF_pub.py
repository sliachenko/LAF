
#
#  Written by Serguei Liachenko
#  Serguei.Liachenko@fda.hhs.gov
#
#  October 2023
#
#  This is the code supplied with the manuscript titled
#  "Fusion of orthogonal MRI scans into isotropic images using
#  linear algebraic approach" as a demonstration of the method
#  
#  It uses (supplied also) three orthogonal projection images
#  AXL, COR, and SAG (raw format 180 x 180 x 60 voxels, 32-bit 
#  float) of the same subject (kiwi fruit) to fuse them into high
#  resolution isotropic image LAF (raw format 180 x 180 x 180 
#  voxels, 32 bit float). 
#  Names of the files could be changed at the user discretion in
#  the code below.
#  
#  This code requires numpy library
#


import numpy as np


# this function translates sagittal projection into axial

def SAG_to_AXL(in_array):
    ar = np.rot90(in_array, k=1, axes=(0,1))
    ar = np.rot90(ar, k=1, axes=(2,1))
    return(ar)

# this function translates coronal projection into axial

def COR_to_AXL(in_array):
    ar = np.rot90(in_array, k=1, axes=(1,0))
    ar = np.flip(ar, 0)    
    return(ar)

# this function creates linear equation coefficient matrix for
# N x N x N system

def create_matrix(N):
    matrix = np.zeros((N**3, N**3))
    for i in range(N**2):
        for k in range(N):
            matrix[i, N*i+k]=1
    for i in range(N):
        for m in range(N):
            for k in range(N):
                matrix[i+m*N+N**2, k*N+i+m*N**2] = 1
    for i in range(N**2):
        for k in range(N):
            matrix[i+2*N**2, k*N**2+i] = 1
    return(matrix)

# set dimentionality of fused image (180 x 180 x 180)
# set voxel aspect ratio (3:1)
# set number of slices for orthogonal projections to 60

Dim = 180
Cube = 3
Depth = np.floor_divide(Dim, Cube)

# reads AXL, SAG, and COR projections and translate SAG and COR 
# into AXL
# user can change image locations below

dtype = np.float32
byte_order = 'big'

file_path_AXL = 'AXL'
file_path_COR = 'COR'
file_path_SAG = 'SAG'

with open(file_path_AXL, 'rb') as file:
    raw_data = file.read()
AXL = np.frombuffer(raw_data, dtype=dtype)
AXL = AXL.reshape((Depth, Dim, Dim))

with open(file_path_SAG, 'rb') as file:
    raw_data = file.read()
SAG = np.frombuffer(raw_data, dtype=dtype)
SAG = SAG.reshape((Depth, Dim, Dim))
SAG = SAG_to_AXL(SAG)

with open(file_path_COR, 'rb') as file:
    raw_data = file.read()
COR = np.frombuffer(raw_data, dtype=dtype)
COR = COR.reshape((Depth, Dim, Dim))
COR = COR_to_AXL(COR)

# initialise fused array and create coefficient matrix for 3:1
# voxel aspect ratio system

fused = np.zeros((Dim, Dim, Dim))
mat = create_matrix(Cube)

# For each 3 x 3 x 3 cluster:
# create known vector,
# solve linear system using least squares,
# reshape solution vector into 3 x 3 x 3 isotropic cluster,
# and put it into fused array

for x in range(Depth):
   for y in range(Depth):
      for z in range(Depth):
         known_vec = np.zeros(Cube**3)
         cAXL = AXL[x, y*Cube:y*Cube+Cube, z*Cube:z*Cube+Cube]
         cAXL = cAXL.flatten()
         cCOR = COR[x*Cube:x*Cube+Cube, y, z*Cube:z*Cube+Cube]
         cCOR = cCOR.flatten()
         cSAG = SAG[x*Cube:x*Cube+Cube, y*Cube:y*Cube+Cube, z]
         cSAG = cSAG.flatten()
         kvec = np.concatenate((cSAG, cCOR, cAXL), axis=None)
         sol,r,rnk,res = np.linalg.lstsq(mat, kvec, rcond=None)
         sol = np.reshape(sol, (Cube, Cube, Cube))
         for i in range(Cube):
             for k in range(Cube):
                 for m in range(Cube):
                     fused[x*Cube+i,y*Cube+k,z*Cube+m]=sol[i,k,m]

out_file_path = 'LAF'
fused.astype(dtype).tofile(out_file_path)



