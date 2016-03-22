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
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "org_fairsim_accel_AccelVectorCplx.h"
#include "org_fairsim_accel_AccelVectorCplx2d.h"
#include "stdC.h"


// allocate the vector
JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorCplx_alloc
  (JNIEnv * env, jobject mo, jint len) {

    //float complex * vec = calloc( len, sizeof(float complex));
    float complex * vec = fftwf_alloc_complex( len );
    memset( vec, 0, len*sizeof( float complex));

    //printf("CPLX Vector allocated, size %d\n", len);
    return (jlong)vec;
}

// de-allocate the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_dealloc
  (JNIEnv * env, jobject mo, jlong addr) {
    fftwf_free( (float complex *)addr );
    //printf("CPLX Vector de-allocated\n");
}


// copy our content to java
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_copyBuffer
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
    float complex * native = (float complex *)addr;

    // memcpy
    if ( toJava ) {
	memcpy( java, native, size*sizeof(float complex) );
    } else {
	memcpy( native, java, size*sizeof(float complex) );
    }   
     
    // de-reference java-side array
    (*env)->ReleasePrimitiveArrayCritical(env, javaArr, java, 0);

}

// copy from real vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeCOPYREAL
  (JNIEnv *env, jobject mo , jlong vt, jlong v1, jint len) {

    float complex * ft = (float complex *)vt;
    float * f1 = (float *)v1;

#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] = f1[i];
}

JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeCOPYCPLX
  (JNIEnv *env, jobject mo , jlong vt, jlong v1, jint len) {

    float complex * ft = (float complex *)vt;
    float complex * f1 = (float complex *)v1;
    
    memcpy( ft, f1, len*sizeof(float complex));

}


// add vectors
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeAdd
  (JNIEnv * env, jobject mo, jlong vt, jlong v1, jint len) {

    float complex * ft = (float complex *)vt;
    float complex * f1 = (float complex *)v1;
    
#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] += f1[i];

}


// axpy
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeAXPY
  (JNIEnv *env, jobject mo, jfloat re, jfloat im, jlong vt, jlong v1, jint len) {
    
   float complex * ft = (float complex *)vt;
   float complex * f1 = (float complex *)v1;
    float complex scal = re + I*im;    

#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] += scal*f1[i];

}


// times
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeTIMES
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len, jboolean conj) {

   float complex * ft = (float complex *)vt;
   float complex * f1 = (float complex *)v1;
   
if ( !conj ) { 
#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] *= f1[i];
} else {
#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] *= conjf( f1[i] );
}

}

// times real
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeTIMESREAL
  (JNIEnv *env, jobject mo, jlong vt, jlong v1, jint len) {

   float complex * ft = (float complex *)vt;
   float * f1 = (float *)v1;
   
#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] *= f1[i];


}

// zero the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeZero
  (JNIEnv *env, jobject mo, jlong ptr, jint len) {

    memset( (float complex *) ptr, 0, len*sizeof(float complex));
}

// scale the vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeScal
  (JNIEnv *env, jobject mo, jlong ptr, jint len, jfloat re, jfloat im) {

    float complex *ft  = (float complex *)ptr;
    float complex scal = re  + I * im;

#pragma omp parallel for    
    for (int i=0; i<len; i++)
	ft[i] *= scal;
}


//norm2
JNIEXPORT jdouble JNICALL Java_org_fairsim_accel_AccelVectorCplx_nativeREDUCE
  (JNIEnv *env, jobject mo, jlong ptr, jint len, jboolean sqr) {
    
    float complex *ft  = (float complex *)ptr;
    
    double ret=0;

#pragma omp parallel for reduction (+:ret)
    for (int i=0; i<len; i++) {
	double val = creal(ft[i])*creal(ft[i]);
	val += cimag(ft[i])*cimag(ft[i]);
	ret += val;
    }   

    return ret;

} 


// paste freq
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativePasteFreq
  (JNIEnv *env, jobject mo , jlong ptrOut, jint wo, jint ho, jlong ptrIn, jint wi, jint hi) {

    float complex *out  = (float complex *)ptrOut;
    float complex *in  = (float complex *)ptrIn;
    
    memset( (float complex *) ptrOut, 0, wo*ho*sizeof(float complex));
   
#pragma omp parallel for 
    for (int yi = 0 ; yi < hi; yi++ ) 
    for (int xi = 0 ; xi < wi; xi++ ) {
    
	
	// copy input to correct position in output
	if ( xi<wi && yi < hi ) {
	    int xo = (xi<wi/2)?(xi):(xi+wo/2);
	    int yo = (yi<hi/2)?(yi):(yi+ho/2);
	    out[ xo + wo*yo ]  = in[ xi + wi * yi ];
	}

    }   

}



// setter for 2d vector
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeSet
  (JNIEnv * env, jobject mo, jlong ptr, jint x, jint y, jint width, jfloat re, jfloat im) {

    float complex * ft = (float complex *)ptr;
    ft[ x + y*width ]  = re + I*im;

}

// getter for 2d vector
JNIEXPORT jfloatArray JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeGet
  (JNIEnv *env, jobject mo, jlong ptr, jint x, jint y, jint width) {

    float complex * ft = (float complex *)ptr;
    float a[2]; 
    a[0] = crealf( ft[ x + y*width ] );
    a[1] = cimagf( ft[ x + y*width ] );
    
    jfloatArray result = (*env)->NewFloatArray( env, 2 );
    (*env)->SetFloatArrayRegion( env, result, 0, 2, a); 
    return result;

}


// calling fftw
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeFFT
  (JNIEnv *env, jobject mo, jlong planPtr, jlong dataPtr, jboolean inverse) {

    fftwPlans * pl = (fftwPlans *)planPtr;
    float complex * dat = (float complex *) dataPtr;

    if ( !inverse )
	fftwf_execute_dft( pl->forward, dat, dat ); 
    if ( inverse ) {
	fftwf_execute_dft( pl->inverse, dat, dat ); 
	
	// normalize
	#pragma omp parallel for 
	for(int i=0; i<pl->size; i++) 
	    dat[i]/=pl->size;
	
    }

}

// fourier shift
JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorCplx2d_nativeFourierShift
  (JNIEnv *env, jobject mo, jlong ptr, jint N, jdouble kx, jdouble ky) {

    float complex * val = (float complex *)ptr;

#pragma omp parallel for
    for (int y = 0 ; y < N ; y++)
    for (int x = 0 ; x < N ; x++) { 

	float phaVal = (float)( 2 * M_PI * (kx*x+ky*y)/N);
	float co = cos( phaVal );
	float si = sin( phaVal );
	val[ x + N * y ] *= (co + I*si);
    }
  
}
