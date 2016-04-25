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
    cudaMalloc( &buf, size );
    return (jlong)buf;
};

// allocate size bytes of native, pinned host-side memory
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorFactory_nativeAllocMemoryHost
  (JNIEnv *env, jobject, jint size) {
    void * buf;
    cudaMallocHost( &buf, size );
    return (jlong)buf;
};

// initialize on library load
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    cachedJVM = jvm;
    printf("[fairSIM-CUDA]: Library loaded\n");
    return JNI_VERSION_1_6;
}




