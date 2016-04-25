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

package org.fairsim.accel;

import org.fairsim.utils.Tool;

import org.fairsim.linalg.Vec;
import org.fairsim.linalg.Vec2d;

import org.fairsim.linalg.Cplx;

public class TestAccel {

    Vec.Real [] vjr, vcr;
    Vec.Cplx [] vjc, vcc;

    final AccelVectorFactory avf = AccelVectorFactory.getFactory();
    
    final int loopCount    = 100;
    final int loopCountFFT = loopCount/2;

    final int vecSize = 512*512;
    //final int vecSize = 1024*1024;

    private String natVer="n/a";

    public static void main( String [] arg ){
   
	
	if (arg.length<1 || ( !arg[0].equals("CPU") && !arg[0].equals("CUDA"))) {
	    System.out.println("Usage: CPU CUDA");
	    return;
	}

	String wd = System.getProperty("user.dir")+"/accel/";
	System.out.println("loading library from: "+wd);

	String vers=".";

	if (arg[0].equals("CPU")) {
	    System.load(wd+"libstdcimpl.so");
	    vers="nCPU";
	} 
	if (arg[0].equals("CUDA")) {
	    System.load(wd+"libcudaimpl.so");
	    vers="CUDA";
	}

	TestAccel tt = new TestAccel();
    
	tt.natVer = vers;
	tt.initArrays();

	tt.testCopyModes();

	tt.testAddTimes();
	tt.testFft();
	tt.testGetSet();

	System.exit(0);	
    }

    // test copy modes
    public void testCopyModes() {

	AccelVectorFactory  avf = AccelVectorFactory.getFactory();

	AccelVectorReal [] va = new AccelVectorReal[10];
	AccelVectorReal [] vb = new AccelVectorReal[10];

	for (int i=0; i<10; i++) {
	    va[i] = avf.createReal(512*512);
	    vb[i] = avf.createReal(512*512);
	}

	Tool.Timer t0 = Tool.getTimer();
	Tool.Timer t1 = Tool.getTimer();
	Tool.Timer t2 = Tool.getTimer();
    
	// standard copy
	t0.start();
	for (int i=0; i<10; i++) {
	    va[i].syncBuffer();
	    vb[i].syncBuffer();
	    va[i].add(vb[i]);
	    va[i].readyBuffer();
	}
	AccelVectorFactory.nativeSync();
	t0.stop();


	// pinned host memory
	for (int i=0; i<10; i++) {
	    va[i].ourCopyMode = 1;
	    vb[i].ourCopyMode = 1;
	}
	
	t1.start();
	for (int i=0; i<10; i++) {
	    va[i].syncBuffer();
	    vb[i].syncBuffer();
	    va[i].add(vb[i]);
	    va[i].readyBuffer();
	}
	AccelVectorFactory.nativeSync();
	t1.stop();

	
	// buffered + pinned host memory
	for (int i=0; i<10; i++) {
	    va[i].ourCopyMode = 2;
	    vb[i].ourCopyMode = 2;
	}
	
	t2.start();
	for (int i=0; i<10; i++) {
	    va[i].syncBuffer();
	    vb[i].syncBuffer();
	    va[i].add(vb[i]);
	    va[i].readyBuffer();
	}
	AccelVectorFactory.nativeSync();
	t2.stop();

	//if (true) return;

	Tool.trace(" Copy modes: " +t0+t1+t2);
	




    }





    // test / time fft
    public void testFft() {

	long fftwptr = FFTProvider.getOrCreateInstance(512,512);


	Vec2d.Cplx fjv, fcv;
	fjv = Vec2d.createCplx(512,512);
	fcv = avf.createCplx2D(512,512);
	fjv.copy( vjc[0] );	
	fcv.copy( vjc[0] );	
	
	Tool.Timer t1j = Tool.getTimer();
	Tool.Timer t1c = Tool.getTimer();
	
	System.out.println("--- Testing: FFT --- ");//+fftwptr);

	// jvm warmup
	Vec2d.Cplx jvmWarmUp = Vec2d.createCplx(512,512);
	for (int i=0; i<10; i++)
	    jvmWarmUp.fft2d( (i%2==1) );


	// test and time adding vectors
	t1j.start();
	for ( int i=0; i<loopCountFFT; i++ ) {
	    fjv.fft2d( (i%2==1) );
	}
	t1j.stop();
	
	t1c.start();
	for ( int i=0; i<loopCountFFT; i++ ) {
	    fcv.fft2d( (i%2==1) );
	}
	AccelVectorFactory.nativeSync();
	t1c.stop();
	
	// substract
	System.out.println("Norm java / C: " + fjv.norm2()+" "+fcv.norm2());
	fjv.axpy( -1, fcv );
	System.out.println("Norm java - C: " + fjv.norm2());
	System.out.println("Time JAVA: "+t1j );
	System.out.println("Time "+natVer+": "+t1c);

    }



    // test get/set
    public void testGetSet() {

	System.out.println("---- Testing: getters / setters ----");

	Vec2d.Cplx fjv, fcv;
	fjv = Vec2d.createCplx(512,512);
	fcv = avf.createCplx2D(512,512);
	//fjv.copy( vjc[0] );	
	//fcv.copy( vjc[0] );	

	// access tests
	Cplx.Float sum = Cplx.Float.zero();

	for ( int y=0; y<512; y++) 
	for ( int x=0; x<512; x++) {
	    fjv.set( x,y, new Cplx.Float( x,y));
	    fcv.set( x,y, new Cplx.Float( x,y));
	}
	
	System.out.println("Norm java/"+natVer+
	    ": " + fjv.norm2()+" "+fcv.norm2());
	
	for ( int y=0; y<512; y++) 
	for ( int x=0; x<512; x++) {
	    Cplx.Float t1 = fjv.get( x,y);
	    Cplx.Float t2 = fcv.get( x,y);
	    if ( t1.re != x || t1.im != y ) {
		System.out.println("Fail J: "+t1.re+" "+t1.im);
		return;
	    }
	    if ( t2.re != x || t2.im != y ) {
		System.out.println("Fail C: "+t2.re+" "+t2.im);
		return;
	    }
	}

	// timing
	Tool.Timer t1j = Tool.getTimer();
	Tool.Timer t1c = Tool.getTimer();
	
	t1j.start();
	for (int i=0; i<512*512; i++) {
	    Cplx.Float tmp = fjv.get( (i*5+7)%512 , (i*4+2)%512 );
	    fjv.set( (i*12+3)%512 , (i*17+3)%512, tmp);
	}
	t1j.stop();
	
	t1c.start();
	for (int i=0; i<512*512; i++) {
	    Cplx.Float tmp = fcv.get( (i*5+7)%512 , (i*4+2)%512 );
	    fcv.set( (i*12+3)%512 , (i*17+3)%512, tmp);
	}
	AccelVectorFactory.nativeSync();
	t1c.stop();
	
	System.out.println("Norm java/"+natVer+
	    ": " + fjv.norm2()+" "+fcv.norm2());

	fjv.axpy( -1, fcv);
	System.out.println("Norm diff: " + fjv.norm2());
	System.out.println("Time JAVA: "+t1j );
	System.out.println("Time "+natVer+": "+t1c);

    }





    // test / time vector addition
    public void testAddTimes() {
	Tool.Timer t1j = Tool.getTimer();
	Tool.Timer t1c = Tool.getTimer();
	Tool.Timer t2j = Tool.getTimer();
	Tool.Timer t2c = Tool.getTimer();
	Tool.Timer t3j = Tool.getTimer();
	Tool.Timer t3c = Tool.getTimer();
	Tool.Timer t4j = Tool.getTimer();
	Tool.Timer t4c = Tool.getTimer();

	
	System.out.println("--- Testing: Add, AXPY, Times, Norm2 ---");

	// test and time adding vectors
	t1j.start();
	for ( int i=0; i<loopCount; i++ ) {
	    vjr[1].add( vjr[(i%5)+5] );
	    vjc[1].add( vjc[(i%5)+5] );
	}
	t1j.stop();
	
	t1c.start();
	for ( int i=0; i<loopCount; i++ ) {
	    vcr[1].add( vcr[(i%5)+5] );
	    vcc[1].add( vcc[(i%5)+5] );
	}
	AccelVectorFactory.nativeSync();
	t1c.stop();

	// test and time axpy vectors
	t2j.start();
	for ( int i=0; i<loopCount; i++ ) {
	    vjr[2].axpy( (float)i/loopCount, vjr[(i%5)+5] );
	    vjc[2].axpy( new Cplx.Float( (float)i/loopCount, 0.05f), vjc[(i%5)+5] );
	}
	t2j.stop();
	
	t2c.start();
	for ( int i=0; i<loopCount; i++ ) {
	    vcr[2].axpy( (float)i/loopCount, vcr[(i%5)+5] );
	    vcc[2].axpy( new Cplx.Float( (float)i/loopCount, 0.05f ), vcc[(i%5)+5] );
	}
	AccelVectorFactory.nativeSync();
	t2c.stop();

	
	// test and time 'times' vector
	t3j.start();
	for ( int i=0; i<loopCount; i++ ) {
	    vjr[3].times( vjr[(i%5)+5] );
	    vjc[3].times( vjc[(i%5)+5] );
	}
	t3j.stop();
	
	t3c.start();
	for ( int i=0; i<loopCount; i++ ) {
	    vcr[3].times( vcr[(i%5)+5] );
	    vcc[3].times( vcc[(i%5)+5] );
	}
	AccelVectorFactory.nativeSync();
	t3c.stop();

	// time 'norm2' vector
	double [] normJr = new double[ loopCount ];
	double [] normCr = new double[ loopCount ];
	double [] normJc = new double[ loopCount ];
	double [] normCc = new double[ loopCount ];

	t4j.start();
	for ( int i=0; i<loopCount; i++ ) {
	    normJr[i] = vjr[i%5].norm2();
	    normJc[i] = vjc[i%5].norm2();
	}
	t4j.stop();

	t4c.start();
	for ( int i=0; i<loopCount; i++ ) {
	    normCr[i] = vcr[i%5].norm2();
	    normCc[i] = vcc[i%5].norm2();
	}
	AccelVectorFactory.nativeSync();
	t4c.stop();

	// test norms
	double maxR=0, normAtMaxR=1;
	double maxC=0, normAtMaxC=1;
	for (int i=0; i<loopCount; i++) {
	    if (normJr[i]<0.0 || normCr[i] <0.0 || normJc[i]<0.0 || normCc[i]<0.0) {
		System.out.println("Problem: NEGATIVE NORM!");
	    }
	    double diffR = normJr[i]-normCr[i];
	    double diffC = normJc[i]-normCc[i];
	    if (diffR>maxR) {
		maxR = diffR;
		normAtMaxR = normJr[i];
	    }
	    if (diffC>maxC) {
		maxC = diffC;
		normAtMaxC = normJc[i];
	    }
	    //System.out.println("n: i"+i+" "+diffC+" "+normJc[i]+" "+normCc[i]);
	}
	System.out.println(String.format("Max diff norm2 real (abs/norm/rel): %8.5e %8.5e -> %8.5e",
	    maxR,normAtMaxR, maxR/normAtMaxR));
	System.out.println(String.format("Max diff norm2 cplx (abs/norm/rel): %8.5e %8.5e -> %8.5e",
	    maxC,normAtMaxC, maxC/normAtMaxC));



	// substract
	vjr[1].axpy( -1, vcr[1] );
	vjc[1].axpy( -1, vcc[1] );
	vjr[2].axpy( -1, vcr[2] );
	vjc[2].axpy( -1, vcc[2] );
	vjr[3].axpy( -1, vcr[3] );
	vjc[3].axpy( -1, vcc[3] );

	// norms
	double nrAdd   = vjr[1].norm2() / vcr[1].norm2() ;
	double ncAdd   = vjc[1].norm2() / vcc[1].norm2() ;
	double nrAxpy  = vjr[2].norm2() / vcr[2].norm2() ;
	double ncAxpy  = vjc[2].norm2() / vcc[2].norm2() ;
	double nrTimes = vjr[3].norm2() / vcr[3].norm2() ;
	double ncTimes = vjc[3].norm2() / vcc[3].norm2() ;

	System.out.println(String.format("Dev. real/cplx:   add %5.3e %5.3e", nrAdd  ,ncAdd )); 
	System.out.println(String.format("Dev. real/cplx:  axpy %5.3e %5.3e", nrAxpy ,ncAxpy )); 
	System.out.println(String.format("Dev. real/cplx: times %5.3e %5.3e", nrTimes,ncTimes ));

	System.out.println("Time JAVA: "+t1j+" "+t2j+" "+t3j+" "+t4j);
	System.out.println("Time "+natVer+": "+t1c+" "+t2c+" "+t3c+" "+t4c);

    }



    // initialized the test arrays
    public void initArrays() {

	// create arrays
	vjr = Vec.createArrayReal(10,vecSize);
	vcr = new Vec.Real[10];
	for (int i=0; i<10; i++)
	    vcr[i] = avf.createReal(vecSize);

	vjc = Vec.createArrayCplx(10,vecSize);
	vcc = new Vec.Cplx[10];
	for (int i=0; i<10; i++)
	    vcc[i] = avf.createCplx(vecSize);


	// fill with random data
	for (int i=0; i<10; i++) {
	    float [] vjf1 = vjr[i].vectorData();
	    float [] vcf1 = vcr[i].vectorData();
	    float [] vjf2 = vjc[i].vectorData();
	    float [] vcf2 = vcc[i].vectorData();
	    for (int j=0; j<vecSize; j++) {
		vjf1[j] = (float)Math.random();
		vcf1[j] = vjf1[j];
		vjf2[2*j+0] = (float)Math.random();
		vcf2[2*j+0] = vjf2[2*j+0];
		vjf2[2*j+1] = (float)Math.random();
		vcf2[2*j+1] = vjf2[2*j+1];
	    }
	    vjr[i].syncBuffer();
	    vcr[i].syncBuffer();
	    vjc[i].syncBuffer();
	    vcc[i].syncBuffer();
	}



    }




}



