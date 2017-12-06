#include <cuda_runtime.h>
#include <pycuda-complex.hpp>
#include <pycuda-helpers.hpp>
#include <surface_functions.h>
texture< fp_tex_float, cudaTextureType2D, cudaReadModeElementType> myTexture;
surface< void, 2> mySurf;

__global__ void globalRK4( float *phi_new, float *phiK1, float *phiK2, float diffusionC, float coef, float dt, float dx, float dy){

int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
int tid   = gridDim.x * blockDim.x * tid_y + tid_x;
// Other indices
if (tid_x < blockDim.x*gridDim.x-1 && tid_x > 0 ){
  
  if (tid_y < blockDim.y*gridDim.y-1 && tid_y > 0 ){
    
    float laplax = 0.0;
    float laplay = 0.0;
    float idx = 1.0/dx;
    float idy = 1.0/dy;
    // k1_i+1,j
    int tid_aux = gridDim.x * blockDim.x * tid_y + tid_x + 1;
    laplax = phiK1[tid_aux]-2.0*phiK1[tid];
    // k1_i-1,j
    tid_aux = gridDim.x * blockDim.x * tid_y + tid_x - 1;
    laplax += phiK1[tid_aux];
    laplax *= idx*idx;
    // k1_i,j+1
    tid_aux = gridDim.x * blockDim.x * (tid_y+1) + tid_x;
    laplay = phiK1[tid_aux]-2.0*phiK1[tid];
    // k1_i,j-1
    tid_aux = gridDim.x * blockDim.x * (tid_y-1) + tid_x;
    laplay += phiK1[tid_aux];
    laplay *= idy*idy;
    
    phiK2[tid] = (laplax+laplay)*diffusionC;
    phi_new[tid] += (laplax+laplay)*coef*dt*(1.0/6.0)*diffusionC;
    
    
  }
  
}
//else return;

}

__global__ void sharedRK4(float *phi, float *phi_new, float *phiK2, float coefIter, float coef, float diffusionC, float dt, float dx, float dy){
  
  int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
  int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
  int tid   = gridDim.x * blockDim.x * tid_y + tid_x;
  // State
  float state, stateK2;
  state = phi[tid];
  stateK2 = phiK2[tid];
  // Shared
  __shared__ float k1_sh[ blockDim.x ][ blockDim.y ];
  //__syncthreads();
  k1_sh[ threadIdx.x ][ threadIdx.y ] = state + coefIter*dt*stateK2;
  __syncthreads();
  
  // Calculate Laplacian
  if (tid_x < blockDim.x*gridDim.x-1 && tid_x > 0 ){
    if (tid_y < blockDim.y*gridDim.y-1 && tid_y > 0 ){
      
      float lapla=0.0;
      float idx = 1.0/dx;
      float idy = 1.0/dy;
      
      if (threadIdx.x<blockDim.x-1 && threadIdx.x>0){
	  lapla = k1_sh[threadIdx.x+1][ threadIdx.y ]+k1_sh[threadIdx.x-1][ threadIdx.y ] - 2.0*k1_sh[threadIdx.x][ threadIdx.y ];
	  }
      else if (threadIdx.x == blockDim.x-1) lapla = (phi[tid+1]+ coefIter*dt*phiK2[tid+1]) +k1_sh[threadIdx.x-1][ threadIdx.y ] - 2.0*k1_sh[threadIdx.x][ threadIdx.y ];
      else if (threadIdx.x == 0)            lapla = k1_sh[threadIdx.x+1][ threadIdx.y ]+ (phi[tid-1]+ coefIter*dt*phiK2[tid-1]) - 2.0*k1_sh[threadIdx.x][ threadIdx.y ];
      lapla *= idx*idx;
      
      if (threadIdx.y<blockDim.y-1 && threadIdx.y>0){
	  lapla += (k1_sh[threadIdx.x][ threadIdx.y+1 ]+k1_sh[threadIdx.x][ threadIdx.y-1 ]- 2.0*k1_sh[threadIdx.x][ threadIdx.y ])*idy*idy;}
      else if (threadIdx.y == blockDim.y-1) lapla += ((phi[gridDim.y * blockDim.y * (tid_y+1) + tid_x]+ coefIter*dt*phiK2[gridDim.x * blockDim.x * (tid_y+1) + tid_x])+k1_sh[threadIdx.x][ threadIdx.y-1 ]- 2.0*k1_sh[threadIdx.x][ threadIdx.y ])*idy*idy;
      else if (threadIdx.y == 0)            lapla += (k1_sh[threadIdx.x][ threadIdx.y+1 ] + (phi[gridDim.x * blockDim.x * (tid_y-1) + tid_x]+ coefIter*dt*phiK2[gridDim.x * blockDim.x * (tid_y-1) + tid_x])- 2.0*k1_sh[threadIdx.x][ threadIdx.y ])*idy*idy;

      lapla *= diffusionC;
      phiK2[tid] = lapla;
      phi_new[tid] += coef*dt*(1.0/6.0)*lapla;
    }}
  
  
}

__global__ void textureRK4(float *phi, float *phi_new, float diffusionC, float coef, float coefIter, float dt, float dx, float dy, int lastStep){
  
  int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
  int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
  int tid   = gridDim.x * blockDim.x * tid_y + tid_x;
  float center, left, right;
  float up, down;
  center = fp_tex2D(myTexture, (float)tid_x,   (float)tid_y);
  left   = fp_tex2D(myTexture, (float)tid_x-1, (float)tid_y);
  right  = fp_tex2D(myTexture, (float)tid_x+1, (float)tid_y);
  up     = fp_tex2D(myTexture, (float)tid_x,   (float)tid_y+1);
  down   = fp_tex2D(myTexture, (float)tid_x,   (float)tid_y-1);
  
  float lapla=0.0;
  float idx=1.0/dx;
  float idy=1.0/dy;
  lapla = (left+right-2.0*center)*idx*idx+(up+down-2.0*center)*idy*idy;
  lapla *= diffusionC;
  
  // Boundary
  if (tid_x == blockDim.x*gridDim.x-1 or tid_x == 0 or tid_y == blockDim.y*gridDim.y-1 or tid_y == 0) lapla = 0.0;
  float phi_o = phi[tid];
  float k1_next = phi_o + coefIter*dt*lapla;
  surf2Dwrite( k1_next , mySurf, tid_x*sizeof(float), tid_y,  cudaBoundaryModeClamp);
  
  phi_new[tid] += coef*dt*(1.0/6.0)*lapla;
  
  if (lastStep == 1){
    phi[tid] = phi_new[tid];
    surf2Dwrite( phi_o , mySurf, tid_x*sizeof(float), tid_y,  cudaBoundaryModeClamp);
  }
  
}