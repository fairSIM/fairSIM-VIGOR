.PHONY: clean headerfiles default

default: cuda

ifeq ($(OS), Windows_NT)

DLL=dll
OBJ=obj
INCL=-I /c/Program\ Files/Java/jdk1.8.0_231/include/ -I /c/Program\ Files/Java/jdk1.8.0_231/include/win32

else

DLL=so
OBJ=o
INCL=-I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux 

endif

cuda: libcudaimpl.$(DLL)
cpu: libstdcimpl.$(DLL)

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

    
%.$(OBJ): %.cu cuda_common.h $(headers)
	nvcc -O3 \
	$(INCL) \
	-arch compute_50 \
	-D_FORCE_INLINES \
	--compiler-options '-fPIC -Wall' -c -o $@ $< 


libstdcimpl.$(DLL): $(headers)
	gcc -O3 -march=native \
	$(INCL) \
	-fopenmp -std=c99 -o $@ -shared -fPIC -Wall \
	stdC_real.c stdC_cplx.c stdC_fft.c -lc -lfftw3f

libcudaimpl.$(DLL): cuda_mgmt.$(OBJ) cuda_real.$(OBJ) cuda_cplx.$(OBJ)
	nvcc -O3 \
	$(INCL) \
	-arch compute_50 \
	-D_FORCE_INLINES \
	--compiler-options '-fPIC -Wall'  -shared -o $@ -lcufft \
	$^ 	

	
	
clean:
	rm -f org_fairsim_accel*.h *.o *.so *.obj *.dll


