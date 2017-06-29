.PHONY: clean headerfiles default

default: cuda
 
cuda: libcudaimpl.so
cpu: libstdcimpl.so

headers=org_fairsim_accel_AccelVectorReal.h org_fairsim_accel_AccelVectorReal2d.h org_fairsim_accel_AccelVectorCplx.h org_fairsim_accel_AccelVectorCplx2d.h \
    org_fairsim_accel_FFTProvider.h org_fairsim_accel_AccelVectorFactory.h 

org_fairsim_accel_AccelVectorReal.h: ../org/fairsim/accel/AccelVectorReal.java
	javah -cp ../ org.fairsim.accel.AccelVectorReal
	touch $@

org_fairsim_accel_AccelVectorReal2d.h: ../org/fairsim/accel/AccelVectorReal.java
	javah -cp ../ org.fairsim.accel.AccelVectorReal2d
	touch $@

org_fairsim_accel_AccelVectorCplx.h:   ../org/fairsim/accel/AccelVectorCplx.java
	javah -cp ../ org.fairsim.accel.AccelVectorCplx
	touch $@

org_fairsim_accel_AccelVectorCplx2d.h: ../org/fairsim/accel/AccelVectorCplx.java
	javah -cp ../ org.fairsim.accel.AccelVectorCplx2d
	touch $@

org_fairsim_accel_FFTProvider.h: ../org/fairsim/accel/FFTProvider.java
	javah -cp ../ org.fairsim.accel.FFTProvider
	touch $@

org_fairsim_accel_AccelVectorFactory.h: ../org/fairsim/accel/AccelVectorFactory.java
	javah -cp ../ org.fairsim.accel.AccelVectorFactory
	touch $@

headerfiles: $(headers)

    
%.o: %.cu cuda_common.h $(headers)
	nvcc -O3 \
	-I/usr/lib/jvm/java-8-openjdk-amd64/include \
	-I/usr/lib/jvm/java-8-openjdk-amd64/include/linux \
	-arch compute_20 \
	-D_FORCE_INLINES \
	--compiler-options '-fPIC -Wall' -c -o $@ $< 


libstdcimpl.so: $(headers)
	gcc -O3 -march=native \
	-I/usr/lib/jvm/java-8-openjdk-amd64/include \
	-I/usr/lib/jvm/java-8-openjdk-amd64/include/linux \
	-fopenmp -std=c99 -o $@ -shared -fPIC -Wall \
	stdC_real.c stdC_cplx.c stdC_fft.c -lc -lfftw3f

libcudaimpl.so: cuda_mgmt.o cuda_real.o cuda_cplx.o 
	nvcc -O3 \
	-I/usr/lib/jvm/java-8-openjdk-amd64/include \
	-I/usr/lib/jvm/java-8-openjdk-amd64/include/linux \
	-arch compute_20 \
	-D_FORCE_INLINES \
	--compiler-options '-fPIC -Wall'  -shared -o $@ -lcufft \
	cuda_mgmt.o cuda_real.o cuda_cplx.o

	
	
clean:
	rm -f org_fairsim_accel*.h *.o *.so

