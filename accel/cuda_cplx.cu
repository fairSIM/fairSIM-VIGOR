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
#include <ccomplex>
#include <stdint.h>

#include <cufft.h>
#include <cuComplex.h>

#include "org_fairsim_accel_AccelVectorReal.h"
#include "org_fairsim_accel_AccelVectorReal2d.h"
#include "org_fairsim_accel_AccelVectorCplx.h"
#include "org_fairsim_accel_AccelVectorCplx2d.h"
#include "org_fairsim_accel_AccelVectorFactory.h"
#include "org_fairsim_accel_FFTProvider.h"
#include "cuda_common.h"

// =================== COMPLEX VECTORS ===============================

// allocate the vector
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorCplx_alloc
  (JNIEnv * env, jobject mo, jobject factory, jint len) {

    const int maxReduceBlocks = (len+nrReduceThreads-1)/nrReduceThreads;

    cplxVecHandle * vec = (cplxVecHandle *)calloc(1, sizeof(cplxVecHandle));
    vec->len = len;
    vec->size = len*sizeof(cuComplex);

    if (cudaRE( cudaMalloc( (void**)&vec->data,	len*sizeof(cuComplex)))) 
	return 0;
    
    cudaRE( cudaMemset( (cuComplex *)vec->data, 0,	len*sizeof(cuComplex)));

    if (cudaRE( cudaMalloc(  (void**)&vec->deviceReduceBuffer,  sizeof(cuComplex)*maxReduceBlocks )))
	return 0 ;
    if ( cudaRE( cudaMallocHost((void**)&vec->hostReduceBuffer,  sizeof(cuComplex)*maxReduceBlocks )) )
	return 0 ; 

    cudaRE( cudaStreamCreate( &vec->vecStream ) );
    
    // store link to the vector factory
    jclass avfCl  = env->GetObjectClass( factory );
    
    vec->factoryClass = (jclass)env->NewGlobalRef( avfCl );
    vec->factoryInstance    =   env->NewGlobalRef( factory ); 
    
    vec->retBufHost = env->GetMethodID( vec->factoryClass, "returnNativeHostBuffer", "(J)V");
    vec->retBufDev  = env->GetMethodID( vec->factoryClass, "returnNativeDeviceBuffer", "(J)V");
 
    return (jlong)vec;
}

// de-allocate the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_dealloc
  (JNIEnv * env, jobject mo, jlong addr) {

    cplxVecHandle * vec = (cplxVecHandle *)addr;
    cudaRE( cudaStreamSynchronize( vec->vecStream));
    cudaRE( cudaStreamDestroy( vec->vecStream ));

    cudaRE( cudaFree( vec->data ));
    cudaRE( cudaFree( vec->deviceReduceBuffer ));
    cudaRE( cudaFreeHost( vec->hostReduceBuffer ));
    
    env->DeleteGlobalRef( vec->factoryClass );
    env->DeleteGlobalRef( vec->factoryInstance );
    
    free( vec );
}

/** Copy content JAVA <-> GPU memory. This can be done in 3 versions:
 *  0) standard cudaMemcpy (syncronous)
 *  1) cudaMemcpyAsync, with HostRegistered memory 
 *  2) cudaMemcpyAsync, with copy to pinned buffers
 *  -> see Memory-ReadMe, on pro/cons
 * */
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_copyBuffer
  (JNIEnv *env, jobject mo, jlong addr, jfloatArray javaArr, 
    jlong buffer, jboolean toJava, jint copyMode) {

    // get the vector to act on
    cplxVecHandle * native = (cplxVecHandle *)addr;

    // get the java-side buffer
    jfloat * java  = (jfloat *)env->GetPrimitiveArrayCritical( javaArr, 0);
    if ( java == NULL ) {
	jclass exClass = env->FindClass( "java/lang/OutOfMemoryError" );
	env->ThrowNew( exClass, "JNI Buffer copy OOM");
    }	    

    // sync copy, blocks both this thread and full GPU
    if ( copyMode == 0 ) {
	if (!toJava) {
	    cudaRE( cudaMemcpy( native->data, java, native->size, cudaMemcpyHostToDevice) );
	} else {
	    cudaRE( cudaMemcpy( java, native->data, native->size, cudaMemcpyDeviceToHost) );
	}
	env->ReleasePrimitiveArrayCritical( javaArr, java, native->size );  // unlock java memory
    }
    
    // Async copy by pinning the java-provided memory (doesn't block GPU, but this thread)
    if ( copyMode == 1 ) {
	
	cudaRE( cudaHostRegister( java, native->size, 0));	// register, start copy
	if ( !toJava ) {
	    cudaRE( cudaMemcpyAsync( native->data, java, native->size, 
		cudaMemcpyHostToDevice, native->vecStream) );
	} else {
	    cudaRE( cudaMemcpyAsync( java, native->data, native->size, 
		cudaMemcpyDeviceToHost, native->vecStream) );
	}
	cudaRE( cudaStreamSynchronize( native->vecStream ) );	// wait for copy to complete
	cudaRE( cudaHostUnregister( java ) );			// unregister the memory
	env->ReleasePrimitiveArrayCritical( javaArr, java, native->size );  // unlock java memory
    }

    // Async copy by copying to buffer first (blocks neither this thread nor GPU)
    if ( copyMode == 2 ) {

	if ( buffer == 0 ) {
	    fprintf(stderr, "Null pointer (in copy mode 2)!\n");
	    return;
	}   
	
	native->tmpHostBuffer = (void*)buffer;

	// to GPU, fully async
	if ( !toJava ) {
	    // copy to host-side pinned buffer
	    memcpy( native->tmpHostBuffer, java, native->size );   
	    env->ReleasePrimitiveArrayCritical( javaArr, java, native->size );  
	
	    // start async transfer to device
	    cudaRE( cudaMemcpyAsync( native->data, native->tmpHostBuffer, native->size,
		cudaMemcpyHostToDevice, native->vecStream));

	    // add callback to give back the buffer after transfer
	    cudaRE( cudaStreamAddCallback( native->vecStream, 
		&returnCplxBufferToJava, (void*)native, 0)); 
	}
	// to CPU, async but have to wait since when we return data should be there
	if ( toJava ) {
	    // start async transfer from device
	    cudaRE( cudaMemcpyAsync( native->tmpHostBuffer, native->data, native->size,
		cudaMemcpyDeviceToHost, native->vecStream) );
	    
	    // wait for copy to complete
	    cudaRE( cudaStreamSynchronize( native->vecStream ) );	

	    // copy to java, release array, return buffer
	    memcpy( java, native->tmpHostBuffer, native->size );   
	    env->ReleasePrimitiveArrayCritical( javaArr, java, native->size );  
	    env->CallVoidMethod( native->factoryInstance, native->retBufHost, native->tmpHostBuffer);
	}
    
    }

}

// ---- LinAlg functions ----

// COPY from real-valued vector (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeCOPYREAL
  (JNIEnv *env, jobject mo , jlong vt, jlong v1, jint len) {
    
    cplxVecHandle * ft = (cplxVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    // wait for everything in f1 to complete
    syncStreams( ft->vecStream , f1->vecStream);

    // copy 
    kernelCplxCopyReal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads, 
	0, ft->vecStream >>>(len, ft->data, f1->data);

    // let everything in f1 wait for us to complete
    syncStreams( f1->vecStream, ft->vecStream );
}

// COPY from coplex-valued vector (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeCOPYCPLX
  (JNIEnv *env, jobject mo , jlong vt, jlong v1, jint len) {
    
    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;
   
    // wait for everything in f1 to complete
    syncStreams( ft->vecStream, f1->vecStream ); 
    // copy 
    cudaMemcpyAsync( ft->data, f1->data, ft->size, cudaMemcpyDeviceToDevice, ft->vecStream);
    // let everything in f1 wait for us to complete
    syncStreams( f1->vecStream, ft->vecStream ); 

}


// copy short [] to the GPU for direct image processing
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeCOPYSHORT
  (JNIEnv *env, jobject, jlong vt, jlong bufHost, jlong bufDevice, jshortArray javaArr, 
    jint len, jint mode) {
    
    // get the GPU-sided vector
    cplxVecHandle * ft = (cplxVecHandle *)vt;
   
    ft->tmpHostBuffer = (void*)bufHost;
    ft->tmpDevBuffer  = (void*)bufDevice;
 
    // get the java-side buffer
    jshort * java  = (jshort *)(env)->GetPrimitiveArrayCritical(javaArr, 0);
    if ( java == NULL ) {
	jclass exClass = (env)->FindClass( "java/lang/OutOfMemoryError" );
	env->ThrowNew( exClass, "JNI Buffer copy OOM");
    }	    
 
    // copy to pinned host memory
    memcpy( (void*)ft->tmpHostBuffer, java, len*sizeof(uint16_t) );
    
    // de-reference java-side array
    env->ReleasePrimitiveArrayCritical(javaArr, java, 0);

    // copy pinned host to device
    if ( mode == 2 ) { 
	cudaRE( cudaMemcpyAsync( ft->tmpDevBuffer, ft->tmpHostBuffer, 
	    len*sizeof(uint16_t), cudaMemcpyHostToDevice, ft->vecStream ) );
    } 
    else {
	cudaRE( cudaMemcpy( ft->tmpDevBuffer, ft->tmpHostBuffer, 
	    len*sizeof(uint16_t), cudaMemcpyHostToDevice ) );
    }

    // return the host buffer (via callback in stream)
    cudaRE( cudaStreamAddCallback( ft->vecStream, &returnCplxBufferToJava, (void*)ft, 0)); 
    
    // convert short -> float on the GPU
    kernelCplxCopyShort<<<(len+nrCuThreads-1)/nrCuThreads, nrCuThreads,	
	0, ft->vecStream>>>( len, ft->data, (uint16_t*)ft->tmpDevBuffer ) ;

    // return the device buffer (via callback in stream)
    cudaRE( cudaStreamAddCallback( ft->vecStream, &returnCplxDeviceBufferToJava, (void*)ft, 0)); 
};


// Add vectors (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeAdd
  (JNIEnv * env, jobject mo, jlong vt, jlong v1, jint len) {

    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;
    
    syncStreams( ft->vecStream, f1->vecStream );

    kernelCplxAdd<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads, 
	0, ft->vecStream >>>( len, ft->data, f1->data );
    
    syncStreams( f1->vecStream, ft->vecStream );

}

// add constant (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeADDCONST
  (JNIEnv *env, jobject mo, jlong vt, jint len, jfloat areal, jfloat aimag) {

    cuComplex constant = make_cuComplex( areal, aimag);
    cplxVecHandle * ft = (cplxVecHandle *)vt;
    
    kernelCplxAddConst<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads,
	0, ft->vecStream >>>( len, ft->data, constant );
}


// axpy (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeAXPY
  (JNIEnv *env, jobject mo, jfloat re, jfloat im, jlong vt, jlong v1, jint len) {
    
    cuComplex fac = make_cuComplex( re, im);
    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;
    
    syncStreams( ft->vecStream, f1->vecStream );

    kernelCplxAxpy<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads,
	0, ft->vecStream >>>( len, ft->data, f1->data, fac );
    
    syncStreams( f1->vecStream, ft->vecStream );
}

// times (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeTIMES
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len, jboolean conj) {

    cplxVecHandle * ft = (cplxVecHandle *)vt;
    cplxVecHandle * f1 = (cplxVecHandle *)v1;
    
    syncStreams( ft->vecStream, f1->vecStream );

    kernelCplxTimesCplx<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads,
	0, ft->vecStream >>>( len, ft->data, f1->data, conj );
    
    syncStreams( f1->vecStream, ft->vecStream );
}

// times real (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeTIMESREAL
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len) {

    cplxVecHandle * ft = (cplxVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;

    syncStreams( ft->vecStream, f1->vecStream );
    kernelCplxTimesReal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads, 
	0, ft->vecStream >>>( len, ft->data, f1->data );
    syncStreams( f1->vecStream, ft->vecStream );

}



// norm2 (async)
JNIEXPORT jdouble JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeREDUCE
  (JNIEnv *env, jobject mo, jlong vt, jint len, jboolean sqr) {

    const int blocksize = (len+nrReduceThreads-1)/nrReduceThreads;

    //printf("Blocksize: %d", blocksize);

    cplxVecHandle * ft   = (cplxVecHandle *)vt;
    
    kernelCplxNorm2 <<< blocksize, nrReduceThreads, nrReduceThreads*sizeof(cuComplex), 
	ft->vecStream >>>( ft->data, (float*)(ft->deviceReduceBuffer), len, true );

 
    cudaRE( cudaMemcpyAsync( ft->hostReduceBuffer, ft->deviceReduceBuffer, 
	blocksize*sizeof(float), cudaMemcpyDeviceToHost, ft->vecStream ));
    
    cudaRE( cudaStreamSynchronize( ft->vecStream ));    

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

__global__ void kernelCplxCopyShort( int len, cuComplex * out, uint16_t * in ) {
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

__global__ void kernelCplxAddConst( int len, cuComplex *out, cuComplex a) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] = cuCaddf( out[i], a);
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



// zero the vector (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeZero
  (JNIEnv *env, jobject mo, jlong ptr, jint len) {

    cplxVecHandle * ft = ((cplxVecHandle*)ptr);
    cudaRE( cudaMemsetAsync( ft->data, 0, len*sizeof(cuComplex), ft->vecStream ));
}

// scale the vector (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeScal
  (JNIEnv *, jobject mo, jlong ptr, jint len, jfloat re, jfloat im) {

    cplxVecHandle * ft = ((cplxVecHandle*)ptr);
    cuComplex scal = make_cuComplex( re, im );
    
    kernelCplxScal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads,0, ft->vecStream >>>( len, ft->data, scal );
}




// ---- Getters / Setters ----

// setter for 2d vector TODO: not sure if still needed
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeSet
  (JNIEnv * env, jobject mo, jlong ptr, jint x, jint y, jint width, jfloat re, jfloat im) {

    cuComplex * ft = ((cplxVecHandle *)ptr)->data + x + y * width;
    cuComplex set = make_cuComplex( re, im );
    cudaRE( cudaMemcpy( ft, &set, sizeof(cuComplex), cudaMemcpyHostToDevice)); 

}

// getter for 2d vector TODO: not sure if still needed
JNIEXPORT jfloatArray JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeGet
  (JNIEnv *env, jobject mo, jlong ptr, jint x, jint y, jint width) {

    cuComplex * ft = ((cplxVecHandle *)ptr)->data + x + y * width;
    cuComplex get;
    cudaRE( cudaMemcpy( &get, ft, sizeof(cuComplex), cudaMemcpyDeviceToHost)); 

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

    // if there are operations pending in ft->vecStream, sync them to the fft stream
    syncStreams( pl->fftStream, ft->vecStream );

    if ( !inverse )
	cufftExecC2C(pl->cuPlan, dat, dat, CUFFT_FORWARD);
    if ( inverse ) {
	cufftExecC2C(pl->cuPlan, dat, dat, CUFFT_INVERSE);
	cuComplex scal = make_cuComplex( 1./pl->size, 0); 
	kernelCplxScal<<< (pl->size+nrCuThreads-1)/nrCuThreads, nrCuThreads, 
	    0, pl->fftStream >>>( pl->size, dat, scal );
	
    }
    // all fft is happening in the pl->fftStream, so wait for an event there
    syncStreams( ft->vecStream, pl->fftStream );
}

// creating FFT plans
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_FFTProvider_nativeCreatePlan2d
  (JNIEnv *env, jclass mo, jint w, jint h) {

    fftPlan * pl = (fftPlan*)calloc(1, sizeof(fftPlan));
    
    cudaRE( cudaStreamCreate( &pl->fftStream ));
    pl->size = w*h;

    printf("Creating FFTW plan %d x %d ... ", w, h);
    fflush(stdout);
    cufftPlan2d( &(pl->cuPlan), w, h, CUFFT_C2C );
    cufftSetStream( pl->cuPlan, pl->fftStream );
    printf(" done.\n");

    return (jlong)pl;
}

// fourier shift
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeFourierShift
  (JNIEnv *env, jobject mo, jlong ptr, jint N, jdouble kx, jdouble ky) {
    
    dim3 blocks(16,16);
    dim3 numBlocks( (N+blocks.x-1) / blocks.x , (N+blocks.y-1) / blocks.y  );

    cplxVecHandle * ft = (cplxVecHandle *)ptr;
    
    kernelCplxFourierShift <<< numBlocks, blocks,0, ft->vecStream >>> ( N , ft->data, kx, ky );

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
  (JNIEnv *env, jobject mo , jlong ptrOut, jint wo, jint ho, jlong ptrIn, jint wi, jint hi, jint xOff, jint yOff) {

    
    dim3 blocks(16,16);
    dim3 numBlocks( (wi+blocks.x-1) / blocks.x , (hi+blocks.y-1) / blocks.y  );

    cplxVecHandle * fo = (cplxVecHandle *)ptrOut;
    cplxVecHandle * fi = (cplxVecHandle *)ptrIn;

    syncStreams( fo->vecStream, fi->vecStream );

    cudaRE( cudaMemsetAsync( fo->data, 0, wo*ho*sizeof(cuComplex), fo->vecStream));
    kernelCplxPasteFreq<<< numBlocks, blocks, 0, fo->vecStream >>>( fo->data, wo, ho, fi->data, wi, hi, xOff, yOff );

    syncStreams( fi->vecStream, fo->vecStream);

}

__global__ void kernelCplxPasteFreq( cuComplex *out, int wo, int ho, cuComplex *in, int wi, int hi, int xOff, int yOff ) {

    int xi = blockIdx.x*blockDim.x + threadIdx.x;
    int yi = blockIdx.y*blockDim.y + threadIdx.y;
	
    // copy input to correct position in output
    if ( xi<wi && yi < hi ) {
	int xo = (xi<wi/2)?(xi):(xi+wo/2);
	int yo = (yi<hi/2)?(yi):(yi+ho/2);
	xo = ( xo + xOff + wo ) % wo;
	yo = ( yo + yOff + ho ) % ho;
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


