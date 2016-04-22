.PHONY: clean

cpu: libstdcimpl.so
cuda: libcudaimpl.so

headers: org_fairsim_accel_AccelVectorReal.h org_fairsim_accel_AccelVectorReal2d.h org_fairsim_accel_AccelVectorCplx.h org_fairsim_accel_AccelVectorCplx2d.h \
    org_fairsim_accel_FFTProvider.h org_fairsim_accel_AccelVectorFactory.h

org_fairsim_accel_AccelVectorReal.h: ../org/fairsim/accel/AccelVectorReal.java
	javah -cp ../ org.fairsim.accel.AccelVectorReal

org_fairsim_accel_AccelVectorReal2d.h: ../org/fairsim/accel/AccelVectorReal.java
	#rm -f org_fairsim_accel_AccelVectorReal2d.h
	javah -cp ../ org.fairsim.accel.AccelVectorReal2d

org_fairsim_accel_AccelVectorCplx.h:   ../org/fairsim/accel/AccelVectorCplx.java
	javah -cp ../ org.fairsim.accel.AccelVectorCplx

org_fairsim_accel_AccelVectorCplx2d.h: ../org/fairsim/accel/AccelVectorCplx.java
	#rm -f org_fairsim_accel_AccelVectorCplx2d.h
	javah -cp ../ org.fairsim.accel.AccelVectorCplx2d

org_fairsim_accel_FFTProvider.h: ../org/fairsim/accel/FFTProvider.java
	javah -cp ../ org.fairsim.accel.FFTProvider

org_fairsim_accel_AccelVectorFactory.h: ../org/fairsim/accel/AccelVectorFactory.java
	javah -cp ../ org.fairsim.accel.AccelVectorFactory


libstdcimpl.so: headers 
	gcc -O3 -march=native -I/usr/lib/jvm/java-6-openjdk-amd64/include \
	-fopenmp -std=c99 -o libstdcimpl.so -shared -fPIC -Wall \
	stdC_real.c stdC_cplx.c stdC_fft.c -lc -lfftw3f

libcudaimpl.so: headers
	nvcc -O3 -I/usr/lib/jvm/java-6-openjdk-amd64/include -arch compute_20 \
	--compiler-options '-fPIC -Wall' -shared -o libcudaimpl.so -lcufft \
	cuda_impl.cu
	
clean:
	rm -f org_fairsim_accel*.h *.o *.so

