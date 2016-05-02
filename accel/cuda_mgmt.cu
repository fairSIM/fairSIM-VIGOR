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
#include <cuda_profiler_api.h>

#include "org_fairsim_accel_AccelVectorFactory.h"
#include "org_fairsim_accel_FFTProvider.h"
#include "cuda_common.h"

// =================== MANAGEMENT ===============================

JavaVM* cachedJVM; 

// device-wide sync, for timing purposes
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorFactory_nativeSync
  (JNIEnv *env, jclass) {
    cudaDeviceSynchronize();
}

// allocate size bytes of native device-side memory
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorFactory_nativeAllocMemory
  (JNIEnv *env, jobject, jint size) {
    void * buf;
    cudaRE( cudaMalloc( &buf, size ));
    return (jlong)buf;
};

// allocate size bytes of native, pinned host-side memory
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorFactory_nativeAllocMemoryHost
  (JNIEnv *env, jobject, jint size) {
    void * buf;
    cudaRE( cudaMallocHost( &buf, size ));
    return (jlong)buf;
};

// initialize on library load
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    cachedJVM = jvm;
    printf("[fairSIM-CUDA]: Library loaded\n");
    return JNI_VERSION_1_6;
}

// start the cuda profiler
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorFactory_startProfiler
  (JNIEnv *, jclass) {
    cudaProfilerStart();
};

// stop the cuda profiler
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorFactory_stopProfiler
  (JNIEnv *, jclass) {
    cudaProfilerStop();
};


// sync two streams
void syncStreams( cudaStream_t wait, cudaStream_t signal ) {

    // create the event
    cudaEvent_t syncEvent;
    cudaRE( cudaEventCreateWithFlags( &syncEvent, cudaEventDisableTiming ));
    
    // record the event to stream 'theirs'
    cudaRE( cudaEventRecord( syncEvent, signal ) );

    // let our stream wait for the event to occur
    cudaRE( cudaStreamWaitEvent( wait, syncEvent, 0 ));

    // destroy the event
    cudaRE( cudaEventDestroy( syncEvent ));

}

// executed by the async copy operation, returns the host-side pinned buffer
// back to Java for reuse
void returnRealBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr ) {

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

// executed by the async copy operation, returns the host-side pinned buffer
// back to Java for reuse
void returnRealDeviceBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr ) {

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
    env->CallVoidMethod( vec->factoryInstance, vec->retBufDev, vec->tmpDevBuffer);

    if (detachLater)
	cachedJVM->DetachCurrentThread(); 
}

// executed by the async copy operation, returns the host-side pinned buffer
// back to Java for reuse
void returnCplxBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr ) {

    // retrieve vector    
    cplxVecHandle * vec = (cplxVecHandle*)ptr;

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

// executed by the async copy operation, returns the host-side pinned buffer
// back to Java for reuse
void returnCplxDeviceBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr ) {

    // retrieve vector    
    cplxVecHandle * vec = (cplxVecHandle*)ptr;

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
    env->CallVoidMethod( vec->factoryInstance, vec->retBufDev, vec->tmpDevBuffer);

    if (detachLater)
	cachedJVM->DetachCurrentThread(); 
}




