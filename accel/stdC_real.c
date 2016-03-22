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

#include "org_fairsim_accel_AccelVectorReal.h"
#include "stdC.h"


// allocate the vector
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorReal_alloc
  (JNIEnv * env, jobject mo, jint len) {

    float * vec = calloc( len, sizeof(float));
    //printf("REAL Vector allocated, size %d\n", len);
    return (jlong)vec;
}

// de-allocate the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_dealloc
  (JNIEnv * env, jobject mo, jlong addr) {
    free( (float *)addr );
    //printf("REAL Vector de-allocated\n");
}


// copy our content to java
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_copyBuffer
  (JNIEnv *env, jobject mo, jlong addr, jfloatArray javaArr, 
    jboolean toJava, jint size) {

    // get the java-side buffer
    jfloat * java  = (*env)->GetPrimitiveArrayCritical(env, javaArr, 0);
    if ( java == NULL ) {
	jclass exClass = (*env)->FindClass( env,
	    "java/lang/OutOfMemoryError" );
	(*env)->ThrowNew( env, exClass, "JNI Buffer copy OOM");
    }	    
  
    // get our buffer
    float * native = (float *)addr;

    // memcpy
    if ( toJava ) {
	memcpy( java, native, size*4 );
    } else {
	memcpy( native, java, size*4 );
    }   
     
    // de-reference java-side array
    (*env)->ReleasePrimitiveArrayCritical(env, javaArr, java, 0);

}


// add vectors
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAdd
  (JNIEnv * env, jobject mo, jlong vt, jlong v1, jint len) {

    float * ft = (float *)vt;
    float * f1 = (float *)v1;
    
#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] += f1[i];

}

// axpy
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeAXPY
  (JNIEnv *env, jobject mo, jfloat a, jlong vt, jlong v1, jint len) {
    
   float * ft = (float *)vt;
   float * f1 = (float *)v1;
    
#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] += a*f1[i];
}


// times
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeTIMES
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len) {


   float * ft = (float *)vt;
   float * f1 = (float *)v1;
    
#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] *= f1[i];

}

// reduce
JNIEXPORT jdouble JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeREDUCE
  (JNIEnv *env, jobject mo,  jlong vt, jint len, jboolean sqr) {
   
    float * ft = (float *)vt;
    double res=0;

    if (sqr) {
	#pragma omp parallel for reduction(+:res)   
	for (int i=0; i<len; i++)
	    res += ft[i] * ft[i];
    } else {
	#pragma omp parallel for reduction(+:res)   
	for (int i=0; i<len; i++)
	    res += ft[i];
    }

    return res;
}

// setter for 2d vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeSet
  (JNIEnv * env, jobject mo, jlong ptr, jint x, jint y, jint width, jfloat val) {

    float * ft = (float *)ptr;
    ft[ x + y * width ] = val;

}

// getter for 2d vector
JNIEXPORT jfloat JNICALL Java_org_fairsim_accel_AccelVectorReal2d_nativeGet
  (JNIEnv *env, jobject mo, jlong ptr, jint x, jint y, jint width) {

    float * ft = (float *)ptr;
    return ft[ x  + y*width];
}



// zero the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeZero
  (JNIEnv *env, jobject mo, jlong ptr, jint len) {
    memset( (float *)ptr, 0, len*sizeof(float));
}

// scale the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorReal_nativeScal
  (JNIEnv *env, jobject mo, jlong ptr, jint len, jfloat scal) {

    float * ft = (float *)ptr;

#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] *= scal;

}




