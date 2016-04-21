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

#include "org_fairsim_accel_FFTProvider.h"
#include "org_fairsim_accel_AccelVectorFactory.h"
#include "stdC.h"

JNIEXPORT jlong JNICALL Java_org_fairsim_accel_AccelVectorFactory_nativeAllocMemory
  (JNIEnv *env, jclass obj, jint len) {
    return (jlong)malloc( len );
}

JNIEXPORT jlong JNICALL Java_org_fairsim_accel_FFTProvider_nativeCreatePlan2d
  (JNIEnv *env, jclass mo, jint w, jint h) {

    fftwPlans * pl = calloc(1, sizeof(fftwPlans));

    //float complex * tmp  = fftwf_alloc_complex( w*h );
    float complex * tmp  = fftwf_malloc( w*h*sizeof(float complex) );

    printf("Creating FFTW plan %d x %d ... ", w, h);
    fflush(stdout);
    pl->forward = fftwf_plan_dft_2d( w, h, tmp, tmp, FFTW_FORWARD, FFTW_MEASURE );
    pl->inverse = fftwf_plan_dft_2d( w, h, tmp, tmp, FFTW_BACKWARD, FFTW_MEASURE );
    pl->size = w*h;
    printf(" done.\n");
    fflush(stdout);


    return (jlong)pl;
}


JNIEXPORT void JNICALL Java_org_fairsim_accel_AccelVectorFactory_nativeSync
  (JNIEnv *env, jclass c) {};

