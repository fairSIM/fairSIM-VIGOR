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
#ifndef _INC_FAIRSIMJNICUDA_HEADERS
#define _INC_FAIRSIMJNICUDA_HEADERS

static const int  nrReduceThreads = 128 ;    // <-- 2^n, 1024 max.
static const int  nrCuThreads = 256;

extern JavaVM* cachedJVM;

// inspiered here:
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define cudaRE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      
      char errString[1024];
      sprintf(errString,"JNI-CUDA: %s %s %d", cudaGetErrorString(code), file, line);
	
      // retrieve env
      JNIEnv * env; int detachLater=0;
      int getEnvStat = cachedJVM->GetEnv( (void**)&env, JNI_VERSION_1_6);	
      if ( getEnvStat == JNI_EDETACHED) {
        if (cachedJVM->AttachCurrentThread((void **) &env, NULL) != 0) {
	    fprintf(stderr,"Failed to attached JVM");
	}
	detachLater=1;    
      }       

      jclass exClass = (env)->FindClass( "java/lang/RuntimeException" );
      env->ThrowNew( exClass,errString );
      
      if (detachLater)
	cachedJVM->DetachCurrentThread(); 


      //if (abort) exit(code);
   }
}





// cuFFT plan handling
typedef struct {
    cufftHandle cuPlan;		    // plan for this fft
    cudaStream_t fftStream;	    // CUDA stream for this fft
    int size;
} fftPlan;


// structure for real-valued vectors
typedef struct {
    int len;			    // number of elements in vector
    size_t size;		    // number of bytes in vector
    float * data;		    // pointer to data on device
    
    float * deviceReduceBuffer;	    // buffer for reduce operations (on device)
    float * hostReduceBuffer;	    // buffer for reduce operations (on host)
    cudaStream_t vecStream;	    // CUDA stream of operations on this vector
    
    // java management
    jclass  factoryClass;	    // vector factory (on java side)
    jobject factoryInstance;	    // vector factory (on java side)
    jmethodID retBufHost;	    // buffer return function (in java)
    jmethodID retBufDev;	    // buffer return function (in java)
    void * tmpDevBuffer;	    // pointer to host-sided temporary buffer
    void * tmpHostBuffer;	    // pointer to device-sided temporary buffer
} realVecHandle;

// stucture for complex-valued vectors
typedef struct {
    int len;			    // number of elements in vector
    size_t size;		    // number of bytes in vector
    cuComplex * data;		    // pointer to data on device
    
    void * deviceReduceBuffer;	    // buffer for reduce operations (on device)
    void * hostReduceBuffer;	    // buffer for reduce operations (on host)
    cudaStream_t vecStream;	    // CUDA stream of operations on this vector

    // java management
    jclass  factoryClass;	    // vector factory (on java side)
    jobject factoryInstance;	    // vector factory (on java side)
    jmethodID retBufHost;	    // buffer return function (in java)
    jmethodID retBufDev;	    // buffer return function (in java)
    void * tmpDevBuffer;	    // pointer to host-sided temporary buffer
    void * tmpHostBuffer;	    // pointer to device-sided temporary buffer
} cplxVecHandle;

// callback to return async copy buffer
void returnRealBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr );
// callback to return async copy buffer
void returnRealDeviceBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr );

// callback to return async copy buffer
void returnCplxBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr );
// callback to return async copy buffer
void returnCplxDeviceBufferToJava( cudaStream_t stream, cudaError_t status, void* ptr );



// sync theirs stream to ours
void syncStreams( cudaStream_t wait, cudaStream_t signal );

__global__ void kernelAdd( int len, float * out, float * in ); 
__global__ void kernelAddConst( int len, float * out, float c ); 
__global__ void kernelAxpy( int len, float * out, float * in, float a );
__global__ void kernelTimes( int len, float * out, float * in );
__global__ void kernelRealReduce(float *g_idata, float *g_odata, unsigned int n, const bool sqr);
__global__ void kernelRealCopyShort( int len, float * out, uint16_t * in );

__global__ void kernelCplxCopyReal( int len, cuComplex *out, float *in);
__global__ void kernelCplxCopyShort( int len, cuComplex * out, uint16_t * in );
__global__ void kernelCplxAdd( int len, cuComplex * out, cuComplex * in ); 
__global__ void kernelCplxAddConst( int len, cuComplex * out, cuComplex c ); 
__global__ void kernelCplxAxpy( int len, cuComplex * out, cuComplex * in, cuComplex a );
__global__ void kernelCplxTimesCplx( int len, cuComplex * out, cuComplex * in, bool conj );
__global__ void kernelCplxTimesReal( int len, cuComplex * out, float * in );
__global__ void kernelCplxScal( int len, cuComplex * out, cuComplex scal );
__global__ void kernelCplxNorm2(cuComplex *g_idata, float *g_odata, unsigned int n, const bool sqr);
__global__ void kernelCplxReduce(cuComplex *g_idata, cuComplex *g_odata, unsigned int n, const bool sqr);

__global__ void kernelCplxFourierShift( int N, cuComplex * out, float kx, float ky );
__global__ void kernelCplxPasteFreq( cuComplex *out, int wo, int ho, cuComplex *in, int wi, int hi, int  xOff, int yOff );


#endif
