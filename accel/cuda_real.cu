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

    if ( cudaRE( cudaMalloc( (void**)&vec->data,	len*sizeof(float)) ) )
	return 0 ;
    
    cudaRE( cudaMemset( (float *)vec->data, 0,	len*sizeof(float)) );

    if ( cudaRE( cudaMalloc(  (void**)&vec->deviceReduceBuffer, sizeof(float)*maxReduceBlocks) ) )
	return 0;
    if ( cudaRE( cudaMallocHost((void**)&vec->hostReduceBuffer,  sizeof(float) * maxReduceBlocks )) )
	return 0; 
        
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
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_dealloc
  (JNIEnv * env, jobject mo, jlong addr) {

    realVecHandle * vec = (realVecHandle *)addr;
    cudaRE( cudaStreamSynchronize( vec->vecStream) );    // TODO: this might not be needed here
    cudaRE( cudaStreamDestroy( vec->vecStream ) );    

    cudaRE( cudaFree( vec->data ) );
    cudaRE( cudaFree( vec->deviceReduceBuffer ) );
    cudaRE( cudaFreeHost( vec->hostReduceBuffer ) );
    
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
		&returnRealBufferToJava, (void*)native, 0)); 
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

// ---- Linear algebra ----

// add vectors (async)
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAdd
  (JNIEnv * env, jobject mo, jlong vt, jlong v1, jint len) {

    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    syncStreams( ft->vecStream, f1->vecStream );
    kernelAdd<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads, 
	0, ft->vecStream >>>( len, ft->data, f1->data );
    syncStreams( f1->vecStream, ft->vecStream );
}

// add a constant to the vector 
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeADDCONST
  (JNIEnv *env, jobject mo, jlong vt, jint len, jfloat a) {

    realVecHandle * ft = (realVecHandle *)vt;
    kernelAddConst<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads,
	0, ft->vecStream >>>( len, ft->data, (float)a );
}



// axpy
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAXPY
  (JNIEnv *env, jobject mo, jfloat scal, jlong vt, jlong v1, jint len) {
    
    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    syncStreams( ft->vecStream, f1->vecStream );
    kernelAxpy<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads,
	0, ft->vecStream >>>( len, ft->data, f1->data, scal );
    syncStreams( f1->vecStream, ft->vecStream );
}


// times
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeTIMES
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len) {

    realVecHandle * ft = (realVecHandle *)vt;
    realVecHandle * f1 = (realVecHandle *)v1;
    
    syncStreams( ft->vecStream, f1->vecStream );
    kernelTimes<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads,
	0, ft->vecStream >>>( len, ft->data, f1->data );
    syncStreams( f1->vecStream, ft->vecStream );

}

// copy short
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeCOPYSHORT
  (JNIEnv *env, jobject, jlong vt, jlong bufHost, jlong bufDevice, 
    jshortArray javaArr, jint len) {
    
    // get the GPU-sided vector
    realVecHandle * ft = (realVecHandle *)vt;
 
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
    cudaRE( cudaMemcpyAsync( (void*)ft->tmpDevBuffer, (void*)ft->tmpHostBuffer, 
	len*sizeof(uint16_t), cudaMemcpyHostToDevice, ft->vecStream ));
    
    // return the host buffer (via callback in stream)
    cudaRE( cudaStreamAddCallback( ft->vecStream, &returnRealBufferToJava, (void*)ft, 0)); 

    // convert short -> float on the GPU
    kernelRealCopyShort<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads, 0, ft->vecStream >>>( len, ft->data, (uint16_t*)ft->tmpDevBuffer );

    // return the device buffer (via callback in stream)
    cudaRE( cudaStreamAddCallback( ft->vecStream, &returnRealDeviceBufferToJava, (void*)ft, 0)); 

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
 
    cudaMemcpyAsync( ft->hostReduceBuffer, ft->deviceReduceBuffer, 
	blocksize*sizeof(float), cudaMemcpyDeviceToHost, ft->vecStream );
    
    cudaStreamSynchronize( ft->vecStream );    
    
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

__global__ void kernelAddConst( int len, float * out, float a ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) out[i] += a;
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


// setter for 2d vector TODO: These should be deprecated
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeSet
  (JNIEnv * env, jobject mo, jlong ptr, jint x, jint y, jint width, jfloat val) {

    float * ft = ((realVecHandle *)ptr)->data + x + y * width;
    cudaRE( cudaMemcpy( ft, &val, sizeof(float), cudaMemcpyHostToDevice) ); 

}

// getter for 2d vector TODO: These should be deprecated
JNIEXPORT jfloat JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeGet
  (JNIEnv *env, jobject mo, jlong ptr, jint x, jint y, jint width) {

    float * ft = ((realVecHandle *)ptr)->data + x + y * width;
    float get;
    cudaRE( cudaMemcpy( &get, ft, sizeof(float), cudaMemcpyDeviceToHost) ); 

    return get;
}

// zero the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeZero
  (JNIEnv *env, jobject mo, jlong ptr, jint len) {

    float * ft = ((realVecHandle *)ptr)->data;
    cudaRE( cudaMemset( ft, 0, len*sizeof(float)) );
}

// scale the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeScal
  (JNIEnv *env, jobject mo, jlong ptr, jint len, jfloat scal) {

    float * ft = ((realVecHandle *)ptr)->data;
    kernelScal<<< (len+nrCuThreads-1)/nrCuThreads, nrCuThreads >>>( len, ft, scal );
    //cudaDeviceSynchronize();
}



