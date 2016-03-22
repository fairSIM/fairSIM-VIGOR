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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>

#include <cufft.h>
#include <cuComplex.h>

#include "org_fairsim_accel_AccelVectorReal.h"
#include "org_fairsim_accel_AccelVectorReal2d.h"
#include "org_fairsim_accel_AccelVectorCplx.h"
#include "org_fairsim_accel_AccelVectorCplx2d.h"
#include "org_fairsim_accel_AccelVectorFactory.h"
#include "org_fairsim_accel_FFTProvider.h"
#include "cudaC.h"

// =================== REAL VECTORS ===============================

JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorFactory_nativeSync
  (JNIEnv *env, jclass) {
    cudaDeviceSynchronize();
}

const int  nrReduceThreads = 128;    // <-- 2^n, 1024 max.
const int  nrCuThreads = 128;

// allocate the vector
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorReal_alloc
  (JNIEnv * env, jobject mo, jint len) {

    const int maxReduceBlocks = (len+nrReduceThreads-1)/nrReduceThreads;

    realVecHandle * vec = (realVecHandle *)calloc(1, sizeof(realVecHandle));
    vec->len = len;

    cudaMalloc( (void**)&vec->data,	len*sizeof(float));
    cudaMemset( (float *)vec->data, 0,	len*sizeof(float));

    cudaMalloc(  (void**)&vec->deviceReduceBuffer, sizeof(float)*maxReduceBlocks);
  
    //vec->hostReduceBuffer = (float*)malloc( sizeof(float) * maxReduceBlocks ); 
    cudaMallocHost((void**)&vec->hostReduceBuffer,  sizeof(float) * maxReduceBlocks ); 
        
 
    return (jlong)vec;
}

// de-allocate the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_dealloc
  (JNIEnv * env, jobject mo, jlong addr) {

    realVecHandle * vec = (realVecHandle *)addr;

    cudaFree( vec->data );
    cudaFree( vec->deviceReduceBuffer );
    cudaFreeHost( vec->hostReduceBuffer );
    //free( vec->hostReduceBuffer );
    free( vec );
}


// copy our content to java
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_copyBuffer
  (JNIEnv *env, jobject mo, jlong addr, jfloatArray javaArr, 
    jboolean toJava, jint size) {

    // get the java-side buffer
    jfloat * java  = (jfloat *)env->GetPrimitiveArrayCritical( javaArr, 0);
    if ( java == NULL ) {
	jclass exClass = env->FindClass( "java/lang/OutOfMemoryError" );
	env->ThrowNew( exClass, "JNI Buffer copy OOM");
    }	    
  
    // get our buffer
    realVecHandle * native = (realVecHandle *)addr;

    // memcpy
    if ( toJava ) {
	cudaMemcpy( java, native->data, size*sizeof(float), cudaMemcpyDeviceToHost );
	//printf(" to  CPU: %ld\n", (long)native);
    } else {
	cudaMemcpy( native->data, java, size*sizeof(float), cudaMemcpyHostToDevice );
	//printf("from CPU: %ld\n", (long)native);
    }   
    //fflush(stdout);    
     
    // de-reference java-side array
    env->ReleasePrimitiveArrayCritical( javaArr, java, 0);

}



// add vectors
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAdd
  (JNIEnv * env, jobject mo, jlong vt, jlong v1, jint len) {

    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    kernelAdd<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data );
    //cudaDeviceSynchronize();
}

// axpy
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAXPY
  (JNIEnv *env, jobject mo, jfloat scal, jlong vt, jlong v1, jint len) {
    
    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    kernelAxpy<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data, scal );
    //cudaDeviceSynchronize();
}


// times
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeTIMES
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len) {

    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    kernelTimes<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data );
    //cudaDeviceSynchronize();

}




// norm2
JNIEXPORT jdouble JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeREDUCE
  (JNIEnv *env, jobject mo, jlong vt, jint len, jboolean sqr) {

    const int blocksize = (len+nrReduceThreads-1)/nrReduceThreads;

    //printf("Blocksize: %d", blocksize);

    realVecHandle * ft   = (realVecHandle *)vt;
    
    kernelRealReduce <<< blocksize, nrReduceThreads, nrReduceThreads*sizeof(float) >>>
	( ft->data, ft->deviceReduceBuffer, len, true );
    //kernelRealReduce<<< blocksize, nrThreads >>>( ft, reduceBuffer, len, true );
 
    cudaMemcpy( ft->hostReduceBuffer, ft->deviceReduceBuffer, 
	blocksize*sizeof(float), cudaMemcpyDeviceToHost );
    
    
    double res= 0;

    for (int i=0; i<blocksize; i++) {
	res += ft->hostReduceBuffer[i];
    } 

    return res;

}




__global__ void kernelAdd( int len, float * out, float * in ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] += in[i];
}

__global__ void kernelAxpy( int len, float * out, float * in, float a ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] += a*in[i];
}

__global__ void kernelTimes( int len, float * out, float * in ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] = out[i]*in[i];
}

__global__ void kernelScal( int len, float * out, float scal ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] *= scal;
}


// setter for 2d vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeSet
  (JNIEnv * env, jobject mo, jlong ptr, jint x, jint y, jint width, jfloat val) {

    float * ft = ((realVecHandle *)ptr)->data + x + y * width;
    cudaMemcpy( ft, &val, sizeof(float), cudaMemcpyHostToDevice); 

}

// getter for 2d vector
JNIEXPORT jfloat JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeGet
  (JNIEnv *env, jobject mo, jlong ptr, jint x, jint y, jint width) {

    float * ft = ((realVecHandle *)ptr)->data + x + y * width;
    float get;
    cudaMemcpy( &get, ft, sizeof(float), cudaMemcpyDeviceToHost); 

    return get;
}

// zero the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeZero
  (JNIEnv *env, jobject mo, jlong ptr, jint len) {

    float * ft = ((realVecHandle *)ptr)->data;
    cudaMemset( ft, 0, len*sizeof(float));
}

// scale the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeScal
  (JNIEnv *env, jobject mo, jlong ptr, jint len, jfloat scal) {

    float * ft = ((realVecHandle *)ptr)->data;
    kernelScal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft, scal );
    //cudaDeviceSynchronize();
}


// =================== COMPLEX VECTORS ===============================

// allocate the vector
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorCplx_alloc
  (JNIEnv * env, jobject mo, jint len) {

    const int maxReduceBlocks = (len+nrReduceThreads-1)/nrReduceThreads;

    cplxVecHandle * vec = (cplxVecHandle *)calloc(1, sizeof(cplxVecHandle));
    vec->len = len;

    cudaMalloc( (void**)&vec->data,	len*sizeof(cuComplex));
    cudaMemset( (cuComplex *)vec->data, 0,	len*sizeof(cuComplex));

    cudaMalloc(  (void**)&vec->deviceReduceBuffer, sizeof(cuComplex)*maxReduceBlocks);
  
    //vec->hostReduceBuffer = (float*)malloc( sizeof(float) * maxReduceBlocks ); 
    cudaMallocHost((void**)&vec->hostReduceBuffer,  sizeof(cuComplex) * maxReduceBlocks ); 
 
    return (jlong)vec;


}

// de-allocate the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_dealloc
  (JNIEnv * env, jobject mo, jlong addr) {

    cplxVecHandle * vec = (cplxVecHandle *)addr;

    cudaFree( vec->data );
    cudaFree( vec->deviceReduceBuffer );
    cudaFreeHost( vec->hostReduceBuffer );
    //free( vec->hostReduceBuffer );
    free( vec );
}


// copy our content to java
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_copyBuffer
  (JNIEnv *env, jobject mo, jlong addr, jfloatArray javaArr, 
    jboolean toJava, jint size) {

    // get the java-side buffer
    jfloat * java  = (jfloat *)(env)->GetPrimitiveArrayCritical(javaArr, 0);
    if ( java == NULL ) {
	jclass exClass = (env)->FindClass( "java/lang/OutOfMemoryError" );
	env->ThrowNew( exClass, "JNI Buffer copy OOM");
    }	    
  
    // get our buffer
    cplxVecHandle * native = (cplxVecHandle *)addr;

    // memcpy
    if ( toJava ) {
	cudaMemcpy( java, native->data, size*sizeof(cuComplex), cudaMemcpyDeviceToHost );
	//printf(" to  CPU: %ld\n", (long)native);
    } else {
	cudaMemcpy( native->data, java, size*sizeof(cuComplex), cudaMemcpyHostToDevice );
	//printf("from CPU: %ld\n", (long)native);
    }   
    //fflush(stdout);    
 
    // de-reference java-side array
    env->ReleasePrimitiveArrayCritical(javaArr, java, 0);

}

// ---- LinAlg functions ----

// copy vectors
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeCOPYREAL
  (JNIEnv *env, jobject mo , jlong vt, jlong v1, jint len) {
    cplxVecHandle * ft = (cplxVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;

    kernelCplxCopyReal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>(len, ft->data, f1->data);
}

JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeCOPYCPLX
  (JNIEnv *env, jobject mo , jlong vt, jlong v1, jint len) {
    
    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;
    cudaMemcpy( ft->data, f1->data, len*sizeof(cuComplex), cudaMemcpyDeviceToDevice);
}


// add vectors
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeAdd
  (JNIEnv * env, jobject mo, jlong vt, jlong v1, jint len) {

    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;
    
    kernelCplxAdd<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data );
    //cudaDeviceSynchronize();

}

// axpy
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeAXPY
  (JNIEnv *env, jobject mo, jfloat re, jfloat im, jlong vt, jlong v1, jint len) {
    
    cuComplex fac = make_cuComplex( re, im);
    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;
    

    kernelCplxAxpy<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data, fac );
    //cudaDeviceSynchronize();
}

// times
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeTIMES
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len, jboolean conj) {

    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;

    kernelCplxTimesCplx<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data, conj );
    //cudaDeviceSynchronize();
}

JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeTIMESREAL
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len) {

    cplxVecHandle * ft = (cplxVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;

    kernelCplxTimesReal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data );
    //cudaDeviceSynchronize();

}



// norm2
JNIEXPORT jdouble JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeREDUCE
  (JNIEnv *env, jobject mo, jlong vt, jint len, jboolean sqr) {

    const int blocksize = (len+nrReduceThreads-1)/nrReduceThreads;

    //printf("Blocksize: %d", blocksize);

    cplxVecHandle * ft   = (cplxVecHandle *)vt;
    
    kernelCplxNorm2 <<< blocksize, nrReduceThreads, nrReduceThreads*sizeof(cuComplex) >>>
	( ft->data, (float*)(ft->deviceReduceBuffer), len, true );
    //kernelRealReduce<<< blocksize, nrThreads >>>( ft, reduceBuffer, len, true );
 
    cudaMemcpy( ft->hostReduceBuffer, ft->deviceReduceBuffer, 
	blocksize*sizeof(float), cudaMemcpyDeviceToHost );
    
    
    double res= 0;

    for (int i=0; i<blocksize; i++) {
	res += ((float*)(ft->hostReduceBuffer))[i];

    } 

    return res;

}


__global__ void kernelCplxCopyReal( int len, cuComplex * out, float * in ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) {
    out[i].x = in[i];
    out[i].y = 0;
    }
}

__global__ void kernelCplxAdd( int len, cuComplex * out, cuComplex * in ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] = cuCaddf( out[i], in[i]);
}

__global__ void kernelCplxAxpy( int len, cuComplex * out, cuComplex * in, cuComplex a ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] = cuCaddf( cuCmulf( a, in[i]), out[i]);
}

__global__ void kernelCplxTimesCplx( int len, cuComplex * out, cuComplex * in, bool conj ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len && !conj) out[i] =cuCmulf(  in[i], out[i]);
  if (i < len &&  conj) out[i] =cuCmulf(  cuConjf( in[i] ), out[i]);
}

__global__ void kernelCplxTimesReal( int len, cuComplex * out, float * in ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len ) {
	out[i].x *= in[i];
	out[i].y *= in[i];
    }
}

__global__ void kernelCplxScal( int len, cuComplex * out, cuComplex scal ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] =cuCmulf(  scal, out[i]);
}



// zero the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeZero
  (JNIEnv *env, jobject mo, jlong ptr, jint len) {

    cplxVecHandle * ft = ((cplxVecHandle*)ptr);
    cudaMemset( ft->data, 0, len*sizeof(cuComplex));
}

// scale the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeScal
  (JNIEnv *, jobject mo, jlong ptr, jint len, jfloat re, jfloat im) {

    cplxVecHandle * ft = ((cplxVecHandle*)ptr);
    cuComplex scal = make_cuComplex( re, im );
    
    kernelCplxScal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, scal );
    //cudaDeviceSynchronize();
}




// ---- Getters / Setters ----

// setter for 2d vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeSet
  (JNIEnv * env, jobject mo, jlong ptr, jint x, jint y, jint width, jfloat re, jfloat im) {

    cuComplex * ft = ((cplxVecHandle *)ptr)->data + x + y * width;
    cuComplex set = make_cuComplex( re, im );
    cudaMemcpy( ft, &set, sizeof(cuComplex), cudaMemcpyHostToDevice); 

}

// getter for 2d vector
JNIEXPORT jfloatArray JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeGet
  (JNIEnv *env, jobject mo, jlong ptr, jint x, jint y, jint width) {

    cuComplex * ft = ((cplxVecHandle *)ptr)->data + x + y * width;
    cuComplex get;
    cudaMemcpy( &get, ft, sizeof(cuComplex), cudaMemcpyDeviceToHost); 

    float a[2];
    a[0] = cuCrealf( get );
    a[1] = cuCimagf( get );
    
    jfloatArray result = env->NewFloatArray( 2 );
    env->SetFloatArrayRegion( result, 0, 2, a); 
    return result;

}

// =================== FFT ===============================


// calling fftw
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeFFT
  (JNIEnv *env, jobject mo, jlong planPtr, jlong dataPtr, jboolean inverse) {

    fftPlan * pl = (fftPlan*) planPtr;
    cplxVecHandle * ft = (cplxVecHandle *)dataPtr;
    cuComplex * dat = ft->data;

    if ( !inverse )
	cufftExecC2C(pl->cuPlan, dat, dat, CUFFT_FORWARD);
    if ( inverse ) {
	cufftExecC2C(pl->cuPlan, dat, dat, CUFFT_INVERSE);
	
	cuComplex scal = make_cuComplex( 1./pl->size, 0); 
	kernelCplxScal<<< (pl->size+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( pl->size, dat, scal );
	
    }
    //cudaDeviceSynchronize();

}

// creating FFT plans
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_FFTProvider_nativeCreatePlan2d
  (JNIEnv *env, jclass mo, jint w, jint h) {

    fftPlan * pl = (fftPlan*)calloc(1, sizeof(fftPlan));
    pl->size = w*h;

    printf("Creating FFTW plan %d x %d ... ", w, h);
    fflush(stdout);
    cufftPlan2d( &(pl->cuPlan), w, h, CUFFT_C2C );
    printf(" done.\n");

    return (jlong)pl;
}

// fourier shift
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeFourierShift
  (JNIEnv *env, jobject mo, jlong ptr, jint N, jdouble kx, jdouble ky) {
    
    dim3 blocks(16,16);
    dim3 numBlocks( (N+blocks.x-1) / blocks.x , (N+blocks.y-1) / blocks.y  );

    cplxVecHandle * ft = (cplxVecHandle *)ptr;
    
    kernelCplxFourierShift <<< numBlocks, blocks >>> ( N , ft->data, kx, ky );
    //cudaDeviceSynchronize();

}

// fourier shift kernel
__global__ void kernelCplxFourierShift( int N, cuComplex * out, float kx, float ky ) {

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {

	float phaVal = (float)( 2.f * (float)M_PI * (kx*x+ky*y)/N);
	float co = cos( phaVal );
	float si = sin( phaVal );
	cuComplex fac = make_cuComplex(co,si);
	out[ x + N * y ] = cuCmulf( out[ x + N*y] , fac);
    } 
}


// paste freq
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativePasteFreq
  (JNIEnv *env, jobject mo , jlong ptrOut, jint wo, jint ho, jlong ptrIn, jint wi, jint hi) {

    
    dim3 blocks(16,16);
    dim3 numBlocks( (wi+blocks.x-1) / blocks.x , (hi+blocks.y-1) / blocks.y  );

    cplxVecHandle * fo = (cplxVecHandle *)ptrOut;
    cplxVecHandle * fi = (cplxVecHandle *)ptrIn;

    cudaMemset( fo->data, 0, wo*ho*sizeof(cuComplex));
    kernelCplxPasteFreq<<< numBlocks, blocks >>>( fo->data, wo, ho, fi->data, wi, hi );

}

__global__ void kernelCplxPasteFreq( cuComplex *out, int wo, int ho, cuComplex *in, int wi, int hi ) {

    int xi = blockIdx.x*blockDim.x + threadIdx.x;
    int yi = blockIdx.y*blockDim.y + threadIdx.y;
	
    // copy input to correct position in output
    if ( xi<wi && yi < hi ) {
	int xo = (xi<wi/2)?(xi):(xi+wo/2);
	int yo = (yi<hi/2)?(yi):(yi+ho/2);
	out[ xo + wo*yo ].x  = in[ xi + wi * yi ].x;
	out[ xo + wo*yo ].y  = in[ xi + wi * yi ].y;
    }
}

// ========== REDUCTION KERNELS =======

/*
__device__ double kernelCplxNorm2( int N, cuComplex * in ) {

    __shared__ double sum[ blockDim.x ];
    sum[ threadIdx.x ] =0;
    __syncthreads();
}
*/


__global__ void kernelRealReduce(float *g_idata, float *g_odata, unsigned int n, const bool sqr)
{
    extern __shared__ float sdata[];
    //__shared__ float sdata[512];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    if (sqr) 
    while (i < n )
    {
        mySum += g_idata[i] * g_idata[i];
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            mySum += g_idata[i+blockDim.x] * g_idata[i+blockDim.x];
        i += gridSize;
    }
    
    
    if (!sqr)
    while (i < n )
    {
        mySum += g_idata[i] ;
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            mySum += g_idata[i+blockDim.x] ;
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float* smem = sdata;
        if (blockDim.x >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockDim.x >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockDim.x >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockDim.x >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockDim.x >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockDim.x >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}



__global__ void kernelCplxNorm2(cuComplex *g_idata, float *g_odata, unsigned int n, const bool sqr)
{
    extern __shared__ float sdata[];
    //__shared__ float sdata[512];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n )
    {
        mySum += g_idata[i].x * g_idata[i].x + g_idata[i].y * g_idata[i].y;
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            mySum += g_idata[i+blockDim.x].x * g_idata[i+blockDim.x].x
		    +g_idata[i+blockDim.x].y * g_idata[i+blockDim.x].y;
        i += gridSize;
    }
    
    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float* smem = sdata;
        if (blockDim.x >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockDim.x >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockDim.x >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockDim.x >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockDim.x >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockDim.x >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}






__global__ void kernelCplxReduce(cuComplex *g_idata, cuComplex *g_odata, unsigned int n, const bool sqr)
{
    extern __shared__ cuComplex cdata[];
    //__shared__ float sdata[512];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;

    cuComplex mySum = make_cuComplex(0,0);

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    if (sqr) 
    while (i < n )
    {
        mySum = cuCaddf( mySum, cuCmulf( g_idata[i] , g_idata[i] ));
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            mySum = cuCaddf( mySum, cuCmulf( g_idata[i+blockDim.x] , g_idata[i+blockDim.x] ));
        i += gridSize;
    }
    
    
    if (!sqr)
    while (i < n )
    {
        mySum = cuCaddf( mySum, g_idata[i] );
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            mySum = cuCaddf( mySum, g_idata[i+blockDim.x] );
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory
    cdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockDim.x >= 1024) { if (tid < 512) { cdata[tid] = mySum = cuCaddf( mySum , cdata[tid + 512]); } __syncthreads(); }
    if (blockDim.x >= 512)  { if (tid < 256) { cdata[tid] = mySum = cuCaddf( mySum , cdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256)  { if (tid < 128) { cdata[tid] = mySum = cuCaddf( mySum , cdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128)  { if (tid <  64) { cdata[tid] = mySum = cuCaddf( mySum , cdata[tid +  64]); } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile cuComplex * smem = cdata;
        if (blockDim.x >=  64) { smem[tid].x = mySum.x = mySum.x + smem[tid + 32].x; }
        if (blockDim.x >=  64) { smem[tid].y = mySum.y = mySum.y + smem[tid + 32].y; }
        if (blockDim.x >=  32) { smem[tid].x = mySum.x = mySum.x + smem[tid + 16].x; }
        if (blockDim.x >=  32) { smem[tid].y = mySum.y = mySum.y + smem[tid + 16].y; }
        if (blockDim.x >=  16) { smem[tid].x = mySum.x = mySum.x + smem[tid +  8].x; }
        if (blockDim.x >=  16) { smem[tid].y = mySum.y = mySum.y + smem[tid +  8].y; }
        if (blockDim.x >=   8) { smem[tid].x = mySum.x = mySum.x + smem[tid +  4].x; }
        if (blockDim.x >=   8) { smem[tid].y = mySum.y = mySum.y + smem[tid +  4].y; }
        if (blockDim.x >=   4) { smem[tid].x = mySum.x = mySum.x + smem[tid +  2].x; }
        if (blockDim.x >=   4) { smem[tid].y = mySum.y = mySum.y + smem[tid +  2].y; }
        if (blockDim.x >=   2) { smem[tid].x = mySum.x = mySum.x + smem[tid +  1].x; }
        if (blockDim.x >=   2) { smem[tid].y = mySum.y = mySum.y + smem[tid +  1].y; }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = cdata[0];
}


