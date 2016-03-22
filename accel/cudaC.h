/*
This file is part of Free Analysis and Interactive Reconstruction
for Structured Illumination Microscopy (fairSIM).

fairSIM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

fairSIM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with fairSIM.  If not, see <http://www.gnu.org/licenses/>
*/



typedef struct {
    cufftHandle cuPlan;
    int size;
} fftPlan;

typedef struct {
    int len;
    float * data;
    float * deviceReduceBuffer;
    float * hostReduceBuffer;
} realVecHandle;

typedef struct {
    int len;
    cuComplex * data;
    void * deviceReduceBuffer;
    void * hostReduceBuffer;
} cplxVecHandle;



__global__ void kernelAdd( int len, float * out, float * in ); 
__global__ void kernelAxpy( int len, float * out, float * in, float a );
__global__ void kernelTimes( int len, float * out, float * in );
__global__ void kernelRealReduce(float *g_idata, float *g_odata, unsigned int n, const bool sqr);

__global__ void kernelCplxCopyReal( int len, cuComplex *out, float *in);
__global__ void kernelCplxAdd( int len, cuComplex * out, cuComplex * in ); 
__global__ void kernelCplxAxpy( int len, cuComplex * out, cuComplex * in, cuComplex a );
__global__ void kernelCplxTimesCplx( int len, cuComplex * out, cuComplex * in, bool conj );
__global__ void kernelCplxTimesReal( int len, cuComplex * out, float * in );
__global__ void kernelCplxScal( int len, cuComplex * out, cuComplex scal );
__global__ void kernelCplxNorm2(cuComplex *g_idata, float *g_odata, unsigned int n, const bool sqr);
__global__ void kernelCplxReduce(cuComplex *g_idata, cuComplex *g_odata, unsigned int n, const bool sqr);

__global__ void kernelCplxFourierShift( int N, cuComplex * out, float kx, float ky );
__global__ void kernelCplxPasteFreq( cuComplex *out, int wo, int ho, cuComplex *in, int wi, int hi );



