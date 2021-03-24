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


//import java.util.concurrent.ArrayBlockingQueue;
//import java.util.concurrent.BlockingQueue;

import org.fairsim.linalg.*;

import org.fairsim.utils.Tool;
import org.fairsim.utils.Conf;
import org.fairsim.utils.SimpleMT;

import org.fairsim.accel.AccelVectorFactory;

/** Class to profile the workload of a 2D SIM reconstruction */
public class TestProfileSim  extends Thread {

    final int width, height, nrImgs;
    Vec2d.Cplx otfV, fltV, wien;
    Vec2d.Real damp, apoV;
    short [][] inImgsShrt;

    Vec2d.Cplx fullResult;
    Vec2d.Cplx [] inFFT;
    Vec2d.Cplx [] separate;
    Vec2d.Cplx [] shifted;
    //short [] output;
    Vec2d.Real output;
    
    TestProfileSim(int w, int h) {

	// set some variables
	width=w; height=h;
	nrImgs=128;

	// Setup OTFs, Wiener filter, APO
	otfV = Vec2d.createCplx( width, height );
	fltV = Vec2d.createCplx( width, height );
	damp = Vec2d.createReal( width, height );
	wien = Vec2d.createCplx( 2*width, 2*height );
	apoV = Vec2d.createReal( 2*width, 2*height );

	fillRandom( otfV );
	fillRandom( fltV );
	fillRandom( damp );
	fillRandom( wien );
	fillRandom( apoV );

	otfV.makeCoherent();
	fltV.makeCoherent();
	damp.makeCoherent();
	wien.makeCoherent();
	apoV.makeCoherent();

	fullResult  = Vec2d.createCplx( width*2, height*2);
	inFFT    = Vec2d.createArrayCplx( 3, width, height );
	separate = Vec2d.createArrayCplx( 2, width, height );
	shifted  = Vec2d.createArrayCplx( 2, 2*width, 2*height);
	//output = new short[width*height*2*2];
	output = Vec.getBasicVectorFactory().createReal2D(2*width, 2*height);

	// generate the FFT plans
	inFFT[0].fft2d(true);
	inFFT[0].fft2d(false);
	inFFT[0].zero();
	fullResult.fft2d(true);
	fullResult.fft2d(false);
	fullResult.zero();


	// generate some fake input images
	inImgsShrt = new short[nrImgs][width*height];
	for (int i=0; i<nrImgs; i++)
	    fillRandom( inImgsShrt[i] );
    }

    @Override
    public void run() {
	mimicRecon();
    }



    public void mimicRecon( /*
	final short [][] input,
	final Vec2d.Cplx [] inFFT, 
	final Vec2d.Cplx [] separate, 
	final Vec2d.Cplx [] shifted,
	final Vec2d.Cplx  fullResult,
	final Vec2d.Real output */
	) {

	Tool.Timer t1 = Tool.getTimer();
	
	Vec.syncConcurrent();
	
	AccelVectorFactory.startProfiler();
	
	for (int loop=0; loop<15; loop++) {

	t1.start();
	for (int ang=0; ang<3; ang++) {

	    for (int pha=0; pha<3; pha++) {
		// copy input
		inFFT[pha].setFrom16bitPixels( inImgsShrt[ang*3+pha] );
		// corner dampening
		inFFT[pha].times( damp  );
		// run FFT
		inFFT[pha].fft2d( false );
		// add to band separation
		separate[0].axpy( Cplx.Float.random(), inFFT[pha] );
		separate[1].axpy( Cplx.Float.random(), inFFT[pha] );
	    }

	    // mult w. otf
	    separate[0].timesConj( otfV );
	    separate[1].timesConj( otfV );
	    
	    // position in output vector (w. fast Fourier shift)
	    shifted[0].pasteFreq( separate[0], 0, 0 );
	    separate[1].fft2d( true );
	    separate[1].fourierShift( Math.random(), Math.random() );
	    separate[1].fft2d( false );
	    shifted[1].pasteFreq( separate[1], 20, 30 ); 
	    
	    // add result to fullResult	
	    fullResult.add( shifted[0] ); 
	    fullResult.add( shifted[1] ); 
	    
	} // end direction loop

	fullResult.times( wien );
	fullResult.times( apoV );
	fullResult.fft2d(true);

	// copy back result
	output.copy(fullResult);
	
	t1.stop();
	Tool.trace( "full time: "+t1);
	}
	AccelVectorFactory.stopProfiler();

    }
	    



    /** Start from the command line to run the plugin */
    public static void main( String [] arg ) 
	throws	java.lang.InterruptedException {

	// output usage
	if (arg.length<1) {
	    System.err.println("[one|two]");
	    return;
	}

	// set the accelerator module
	String wd = System.getProperty("user.dir")+"/accel/";
	
	String vers=".";
	boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");

	String dllext = (isWindows)?("dll"):("so");
	
	Tool.trace("loading library from: "+wd+ "("+ ((isWindows)?("win"):("linux"))+")" );

	System.load(wd+"libcudaimpl."+dllext);
	Tool.trace("Running with CUDA support now");
	
	AccelVectorFactory avf = AccelVectorFactory.getFactory();
	
	//avf.setDefaultCopyMode(AccelVectorFactory.DEFAULT_COPY_MODE);
	//avf.setDefaultCopyMode(AccelVectorFactory.HOSTPINNED_COPY_MODE);
	avf.setDefaultCopyMode(AccelVectorFactory.BUFFERED_COPY_MODE);

	Vec.setVectorFactory( avf ); 
	SimpleMT.useParallel(false);

	
	// start the reconstruction loop
	TestProfileSim tps1 = new TestProfileSim(512,512);
	TestProfileSim tps2 = new TestProfileSim(512,512);
	TestProfileSim tps3 = new TestProfileSim(512,512);
	TestProfileSim tps4 = new TestProfileSim(512,512);


	Tool.Timer t1 = Tool.getTimer();
	int count = 15;
	if (arg[0].equals("one")) {
	    tps1.start();
	    tps1.join();
	    
	}
	if (arg[0].equals("two")) {
	    tps1.start();
	    tps2.start();
	    tps3.start();
	    tps4.start();

	    tps1.join();
	    tps2.join();
	    tps3.join();
	    tps4.join();
	    count*=4;
	}
	t1.stop();

	Tool.trace("Reconstructed "+count+" images in "+t1+" : "+
	    String.format(" %7.4f fps", count/t1.msElapsed()*1000.));

    }


    static void fillRandom( Vec2d.Cplx a ) {
	for (int y=0;y<a.vectorHeight();y++)
	for (int x=0;x<a.vectorWidth();x++) {
	    a.set(x,y, Cplx.Float.random());
	}
    }
    
    static void fillRandom( Vec2d.Real a ) {
	for (int y=0;y<a.vectorHeight();y++)
	for (int x=0;x<a.vectorWidth();x++) {
	    a.set(x,y, (float)Math.random());
	}
    }
    
    
    static void fillRandom( short [] a ) {
	for (int i=0; i<a.length; i++)
	    a[i] = (short)(Math.random()*10000);
    }

}


