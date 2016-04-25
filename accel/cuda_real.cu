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
#include <stdint.h>

#include <cufft.h>
#include <cuComplex.h>

#include "org_fairsim_accel_AccelVectorReal.h"
#include "org_fairsim_accel_AccelVectorReal2d.h"
#include "org_fairsim_accel_FFTProvider.h"
#include "cuda_common.h"


// =================== REAL VECTORS ===============================

// allocate the vector
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorReal_alloc
  (JNIEnv * env, jobject mo, jobject factory, jint len) {

    const int maxReduceBlocks = (len+nrReduceThreads-1)/nrReduceThreads;

    realVecHandle * vec = (realVecHandle *)calloc(1, sizeof(realVecHandle));
    vec->len  = len;
    vec->size = len*sizeof(float);

    cudaMalloc( (void**)&vec->data,	len*sizeof(float));
    cudaMemset( (float *)vec->data, 0,	len*sizeof(float));

    cudaMalloc(  (void**)&vec->deviceReduceBuffer, sizeof(float)*maxReduceBlocks);
    cudaMallocHost((void**)&vec->hostReduceBuffer,  sizeof(float) * maxReduceBlocks ); 
        
    cudaStreamCreate( &vec->vecStream );

    // store link to the vector factory
    jclass avfCl  = env->GetObjectClass( factory );
    
    vec->factoryClass = (jclass)env->NewGlobalRef( avfCl );
    vec->factoryInstance    =   env->NewGlobalRef( factory ); 
    
    vec->retBufHost = env->GetMethodID( vec->factoryClass, "returnNativeHostBuffer", "(J)V");
    vec->retBufDev  = env->GetMethodID( vec->factoryClass, "returnNativeDeviceBuffer", "(J)V");
 
    return (jlong)vec;
}

// de-allocate the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_dealloc
  (JNIEnv * env, jobject mo, jlong addr) {

    realVecHandle * vec = (realVecHandle *)addr;

    cudaFree( vec->data );
    cudaFree( vec->deviceReduceBuffer );
    cudaFreeHost( vec->hostReduceBuffer );
    cudaStreamDestroy( vec->vecStream );
    
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
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_copyBuffer
  (JNIEnv *env, jobject mo, jlong addr, jfloatArray javaArr, 
    jlong buffer, jboolean toJava, jint copyMode) {

    // get the vector to act on
    realVecHandle * native = (realVecHandle *)addr;

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

	//memcpy( native->tmpHostBuffer, java, native->size );	    // copy to tmp. host-side buffer
	env->ReleasePrimitiveArrayCritical( javaArr, java, native->size );  // unlock java memory
	
	// start async transfer to device
	cudaMemcpyAsync( native->data, native->tmpHostBuffer, native->size,
	    cudaMemcpyHostToDevice, native->vecStream);

	// add callback to give back the buffer after transfer
	cudaRE( cudaStreamAddCallback( native->vecStream, &returnBufferToJava, (void*)native, 0)); 
    
    }

        

}

// executed by the async copy operation, returns the host-side pinned buffer
// back to Java for reuse
void returnBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr ) {

    // retrieve vector    
    realVecHandle * vec = (realVecHandle*)ptr;

    // retrieve env
    JNIEnv * env; int detachLater=0;
    int getEnvStat = cachedJVM->GetEnv( (void**)&env, JNI_VERSION_1_6);	
    if ( getEnvStat == JNI_EDETACHED) {
	if (cachedJVM->AttachCurrentThread((void **) &env, NULL) != 0) {
	    fprintf(stderr,"Failed to attached JVM");
	}
	detachLater=1;    
    }       

    // call callback
    env->CallVoidMethod( vec->factoryInstance, vec->retBufHost, vec->tmpHostBuffer);

    if (detachLater)
	cachedJVM->DetachCurrentThread(); 
}



// add vectors
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAdd
  (JNIEnv * env, jobject mo, jlong vt, jlong v1, jint len) {

    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    kernelAdd<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data );
}

// axpy
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAXPY
  (JNIEnv *env, jobject mo, jfloat scal, jlong vt, jlong v1, jint len) {
    
    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    kernelAxpy<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data, scal );
}


// times
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeTIMES
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len) {

    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    kernelTimes<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft->data, f1->data );

}

// copy short
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeCOPYSHORT
  (JNIEnv *env, jobject, jlong vt, jlong bufHost, jlong buf, jshortArray javaArr, jint len) {
    
    // get the GPU-sided vector
    realVecHandle * ft = (realVecHandle *)vt;
   
    // get the java-side buffer
    jshort * java  = (jshort *)(env)->GetPrimitiveArrayCritical(javaArr, 0);
    if ( java == NULL ) {
	jclass exClass = (env)->FindClass( "java/lang/OutOfMemoryError" );
	env->ThrowNew( exClass, "JNI Buffer copy OOM");
    }	    
 
    // copy to pinned host memory
    cudaMemcpy( (void*)bufHost, java, len*sizeof(uint16_t), cudaMemcpyHostToHost );
    
    // de-reference java-side array
    env->ReleasePrimitiveArrayCritical(javaArr, java, 0);
    
    // copy pinned host to device
    cudaMemcpyAsync( (void*)buf, (void*)bufHost, len*sizeof(uint16_t), cudaMemcpyHostToDevice, ft->vecStream );

    // convert short -> float on the GPU
    kernelRealCopyShort<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads, 0, ft->vecStream >>>( len, ft->data, (uint16_t*)buf );
};



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


__global__ void kernelRealCopyShort( int len, float * out, uint16_t * in ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) {
    out[i] = in[i];
  }
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



