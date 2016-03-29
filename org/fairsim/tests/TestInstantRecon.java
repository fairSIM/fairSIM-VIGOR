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

package org.fairsim.tests;

import org.fairsim.linalg.*;
import org.fairsim.sim_algorithm.*;

import org.fairsim.network.ImageReceiver;
import org.fairsim.network.ImageWrapper;

import org.fairsim.utils.Tool;
import org.fairsim.utils.Conf;
import org.fairsim.utils.SimpleMT;
import org.fairsim.utils.ImageDisplay;

import org.fairsim.accel.AccelVectorFactory;

/** Class to run instant SIM reconstruction with fixed parameters. */
public class TestInstantRecon  {

   /** Start from the command line to run the plugin */
    public static void main( String [] arg ) {

	// output usage
	if (arg.length<2) {
	    System.err.println("[JAVA|CUDA] (sim-param.xml)");
	    return;
	}

	// set the accelerator module
	boolean set=false;
	String wd = System.getProperty("user.dir")+"/accel/";
	Tool.trace("loading library from: "+wd);

	if (arg[0].equals("CUDA")) {
	    System.load(wd+"libcudaimpl.so");
	    Tool.trace("Running with CUDA support now");
	    Vec.setVectorFactory( AccelVectorFactory.getFactory()); 
	    set=true;
	    SimpleMT.useParallel(false);
	}
	if (arg[0].equals("JAVA")) {
	    set=true;
	}
	if (set==false) {
	    System.err.println("pass either: JAVA, CUDA");
	    return;
	}

	// load the parameter set
	SimParam sp = null;
	try {
	    Conf cfg = Conf.loadFile( arg[1] );
	    sp = SimParam.loadConfig( cfg.r() );
	    OtfProvider otf = OtfProvider.loadFromConfig( cfg );
	    sp.otf( otf );
	    if ( otf == null )
		throw new Exception("No OTF found");

	} catch ( Exception e ) {
	    System.err.println("Failed to load parameters: "+e);
	    return;
	}
	
	// start the reconstruction loop
	networkedReconstruction( sp );


    }


    /** Step-by-step reconstruction process. */
    public static void networkedReconstruction( final SimParam param ) {
	
	final int width=512, height=512;
	param.setPxlSize( width, 0.08 );
	
	final OtfProvider otfPr  = param.otf(); 
	
	double apoStr  = 0.99;
	double apoFWHM = 1.5;
	double wienParam = 0.05;


	// Setup network link
	ImageReceiver ir = new ImageReceiver(50,512,512);
	boolean keepRunning = true;

	ir.addListener( new ImageReceiver.Notify() {
	    public void message( String m , boolean err, boolean fail) {
		String e = (err)?( (fail)?("FAIL: "):("err: ") ):("net: ");
		Tool.trace( e + m );
	    }
	});

	try {
	    ir.startReceiving(null,null);	
	} catch  ( java.io.IOException e ) {
	    Tool.trace("Net setup failed: "+e);
	    return;
	}

	// Setup OTFs, Wiener filter, APO
	Vec2d.Cplx[] otfV    = Vec2d.createArrayCplx( param.nrBand(), 
				param.vectorWidth(), param.vectorHeight() );
	
	for (int i=0; i<param.nrBand(); i++) {
	    otfPr.writeOtfWithAttVector( otfV[i], i, 0,0 );
	    otfV[i].makeCoherent();

	    Tool.trace(String.format("OTF norm2: %7.4e ",otfV[i].norm2()));
	}

	WienerFilter wFilter = new WienerFilter( param );
	Vec2d.Real wienerDenom = wFilter.getDenominator( wienParam );
	wienerDenom.makeCoherent();

	Vec2d.Cplx apoVector = Vec2d.createCplx(2*width,2*height);
	otfPr.writeApoVector( apoVector, apoStr, apoFWHM);
	apoVector.makeCoherent();

	// vectors to store the result
	Vec2d.Cplx fullResult   = Vec2d.createCplx( param, 2);
	Vec2d.Cplx [] separate  = Vec2d.createArrayCplx( param.nrBand()*2-1, width, height);
	Vec2d.Cplx [] shifted	= Vec2d.createArrayCplx( param.nrBand()*2-1, 2*width, 2*height);

	Vec2d.Real tmpImgReal   = Vec2d.createReal(width,height);


	// Big reconstruct loop
	while ( keepRunning ) {
    
	    Tool.Timer t1 = Tool.getTimer();
	    int count=0;

	    // detect white
	    for ( boolean foundSync = false; !foundSync;  ) {
		
		if (count%100==0) t1.start();

		ir.takeImage().writeToVector( tmpImgReal );
		//Tool.trace("Img val: "+tmpImgReal.sumElements() / tmpImgReal.vectorSize());
		
		if (count%100==99) {
		    t1.stop();
		    Tool.trace("100 images took: "+t1);
		}
		count++;
	    } 






	}

    }


	/*
	// Copy current stack into vectors, apotize borders 
	Vec2d.Real [] imgs = new Vec2d.Real[ inSt.getSize() ]; 
	for (int i=0; i<inSt.getSize();i++) { 
	    imgs[i]  = ImageVector.copy( inSt.getProcessor(i+1) );
	    SimUtils.fadeBorderCos( imgs[i] , 10);
	    // for debug, output some information
	    //Tool.trace( String.format("image %2d norm: %15.2f", i, imgs[i].norm2()));
	}

	{
	    Vec2d.Cplx tmpV = Vec2d.createCplx( param );
	    tmpV.fft2d(false);
	}
	

	// compute the input FFT
	Vec2d.Cplx [][] inFFT = new Vec2d.Cplx[ inSt.getSize()/nrPhases ][nrPhases];
	for (int i=0; i<inSt.getSize();i++) { 
		inFFT[i/nrPhases][i%nrPhases] = Vec2d.createCplx( w, h);
	}

	tRec.start();
	tInFft.start();
	for (int i=0; i<inSt.getSize();i++) { 
		inFFT[i/nrPhases][i%nrPhases].copy( imgs[i] );
		Transforms.fft2d( inFFT[i/nrPhases][i%nrPhases] , false);
	}
	Vec.syncConcurrent();
	tInFft.stop();
	tRec.hold();
	*/
    
	/*	
	// loop all pattern directions
	for (int angIdx = 0; angIdx < param.nrDir(); angIdx ++ ) 
	{
	    final SimParam.Dir par = param.dir(angIdx);

	    // ----- Band separation & OTF multiplication (if before shift) -------

	    tBandSep.start();
	    BandSeparation.separateBands( inFFT[angIdx] , separate , 
		    par.getPhases(), par.nrBand(), par.getModulations());
	    Vec.syncConcurrent();
	    tBandSep.hold();
	    
	    Tool.trace(String.format(" SEPR (!otf) 0,1,2: norm2 :: %7.4e %7.4e %7.4e",
		separate[0].norm2(), separate[1].norm2(), separate[3].norm2()));
		

	    tOtfAppl.start();
	    //if (otfBeforeShift)
		for (int i=0; i<(par.nrBand()*2-1) ;i++)  
		    separate[i].timesConj( otfV[ (i+1)/2 ]);
		    //otfPr.applyOtf( separate[i], (i+1)/2);
	    Vec.syncConcurrent();
	    tOtfAppl.hold();
	    
	    Tool.trace(String.format(" SEPR (*otf) 0,1,2: norm2 :: %7.4e %7.4e %7.4e",
		separate[0].norm2(), separate[1].norm2(), separate[3].norm2()));

	    // ------- Shifts to correct position ----------

	    // first, copy to larger vectors

	    tFqPlace.start();
	    // band 0 is DC, so does not need shifting, only a bigger vector
	    SimUtils.placeFreq( separate[0],  shifted[0]);
	    
	    // higher bands need shifting
	    for ( int b=1; b<par.nrBand(); b++) {
		Tool.trace("REC: Dir "+angIdx+": shift band: "+b+" to: "+par.px(b)+" "+par.py(b));
		
		int pos = b*2, neg = (b*2)-1;	// pos/neg contr. to band
		SimUtils.placeFreq( separate[pos] , shifted[pos]);
		SimUtils.placeFreq( separate[neg] , shifted[neg]);
	    }
	    Vec.syncConcurrent();
	    tFqPlace.hold();

	    Tool.trace(String.format(" SHFT (bef.) 0,1,2: norm2 :: %7.4e %7.4e %7.4e",
		shifted[0].norm2(), shifted[1].norm2(), shifted[3].norm2()));

		// then, fourier shift
	    tFqShift.start();	
	    for ( int b=1; b<par.nrBand(); b++) {
		int pos = b*2, neg = (b*2)-1;	// pos/neg contr. to band
		SimUtils.fourierShift( shifted[pos] ,  par.px(b),  par.py(b) );
		SimUtils.fourierShift( shifted[neg] , -par.px(b), -par.py(b) );
	    }
	    Vec.syncConcurrent();
	    tFqShift.hold();
	   
	    Tool.trace(String.format(" SHFT (fftd) 0,1,2: norm2 :: %7.4e %7.4e %7.4e",
		shifted[0].norm2(), shifted[1].norm2(), shifted[3].norm2()));
	    // ------ OTF multiplication or masking ------

	    tOtfAppl.start();
	    /*
	    if (!otfBeforeShift) {
		// multiply with shifted OTF
		for (int b=0; b<par.nrBand(); b++) {
		    // TODO: This will fail (index -1)
		    int pos = b*2, neg = (b*2)-1;	// pos/neg contr. to band
		    otfPr.applyOtf( shifted[pos], b,  par.px(b),  par.py(b) );
		    otfPr.applyOtf( shifted[neg], b, -par.px(b), -par.py(b) );
		}
	    /*
	     } else { */
		// or mask for OTF support
		//TODO: Re-enable masking support
	    /*
		for (int i=0; i<(par.nrBand()*2-1) ;i++)  
		    //wFilter.maskOtf( shifted[i], angIdx, i);
		    otfPr.maskOtf( shifted[i], angIdx, i); */
	   

	   /*
	    // ------ Sum up result ------
	    
	    tLinAlg.start();
	    for (int i=0;i<par.nrBand()*2-1;i++)  
		fullResult.add( shifted[i] ); 
	    Vec.syncConcurrent();
	    tLinAlg.hold();
	
	    Tool.trace(String.format("result norm2: %7.4e", fullResult.norm2()));
	    
	    // ------ Output intermediate results ------
	    
	    if (visualFeedback>0) {
	
		// per-direction results
		Vec2d.Cplx result = Vec2d.createCplx(2*w,2*h);
		for (int i=0;i<par.nrBand()*2-1;i++)  
		    result.add( shifted[i] ); 

		// loop bands in this direction
		for (int i=0;i<par.nrBand();i++) {     

		    // get wiener denominator for (direction, band), add to full denom for this band
		    Vec2d.Real denom = wFilter.getIntermediateDenominator( angIdx, i, wienParam);
		
		    // add up +- shift for this band
		    Vec2d.Cplx thisband   = shifted[i*2];
		    if (i!=0)
			thisband.add( shifted[i*2-1] );
	
		    // output the wiener denominator
		    if (visualFeedback>1) {
			Vec2d.Real wd = denom.duplicate();
			wd.reciproc();
			wd.normalize();
			Transforms.swapQuadrant( wd );
			pwSt2.addImage( wd, String.format(
			    "a%1d: OTF/Wiener band %1d",angIdx,(i/2) ));
		    }
		    
		    // apply filter and output result
		    thisband.times( denom );
		    
		    pwSt2.addImage( SimUtils.pwSpec( thisband ) ,String.format(
			"a%1d: band %1d",angIdx,(i/2)));
		    spSt2.addImage( SimUtils.spatial( thisband ) ,String.format(
			"a%1d: band %1d",angIdx,(i/2)));
		}

		// per direction wiener denominator	
		Vec2d.Real fDenom =  wFilter.getIntermediateDenominator( angIdx, wienParam);	
		result.times( fDenom );
		    
		// output the wiener denominator
		if (visualFeedback>1) {
		    Vec2d.Real wd = fDenom.duplicate();
		    wd.reciproc();
		    wd.normalize();
		    Transforms.swapQuadrant( wd );
		    pwSt2.addImage( wd, String.format(
			"a%1d: OTF/Wiener all bands",angIdx ));
		}
		
		pwSt2.addImage( SimUtils.pwSpec( result ) ,String.format(
		    "a%1d: all bands",angIdx));
		spSt2.addImage( SimUtils.spatial( result ) ,String.format(
		    "a%1d: all bands",angIdx));
	    
		// power spectra before shift
		if (visualFeedback>2) { 
		    for (int i=0; i<(par.nrBand()*2-1) ;i++)  
		    pwSt.addImage( SimUtils.pwSpec( separate[i] ), String.format(
			"a%1d, sep%1d, seperated band", angIdx, i));
		}
	   
	    }


	}   
	
	Tool.trace("Filtering results");
	
	// multiply by wiener denominator
	
	tLinAlg.start();
	fullResult.times(wienerDenom);
	if (visualFeedback>0) {
	    pwSt2.addImage(  SimUtils.pwSpec( fullResult), "full (w/o APO)");
	    spSt2.addImage(  SimUtils.spatial(fullResult), "full (w/o APO)");
	}

	// multiply by apotization vector	
	fullResult.timesConj(apoVector);
	Vec.syncConcurrent();
	tLinAlg.hold();

	Tool.trace("Done, copying results");

	// output full result
	tCpyBack.start();
	spSt2.addImage( SimUtils.spatial( fullResult), "full result");
	tCpyBack.hold();
	
	if (visualFeedback>0) {
	    pwSt2.addImage( SimUtils.pwSpec( fullResult), "full result");
	}

	// Add wide-field for comparison
	if (visualFeedback>=0) {
	    
	    // obtain the low freq result
	    Vec2d.Cplx lowFreqResult = Vec2d.createCplx( param, 2);
	    
	    // have to do the separation again, result before had the OTF multiplied
	    for (int angIdx = 0; angIdx < param.nrDir(); angIdx ++ ) {
		
		final SimParam.Dir par = param.dir(angIdx);
		
		//Vec2d.Cplx [] separate  = Vec2d.createArrayCplx( par.nrComp(), w, h);
		for (int i=0; i<par.nrComp(); i++)
		    separate[i].zero();

		BandSeparation.separateBands( inFFT[angIdx] , separate , 
		    par.getPhases(), par.nrBand(), par.getModulations());

		Vec2d.Cplx tmp  = Vec2d.createCplx( param, 2 );
		SimUtils.placeFreq( separate[0],  tmp);
		lowFreqResult.add( tmp );
	    }	
	    
	    // now, output the widefield
	    if (visualFeedback>0)
		pwSt2.addImage( SimUtils.pwSpec(lowFreqResult), "Widefield" );
	    spSt2.addImage( SimUtils.spatial(lowFreqResult), "Widefield" );
	
	    // otf-multiply and wiener-filter the wide-field
	    otfPr.otfToVector( lowFreqResult, 0, 0, 0, false, false ); 

	    Vec2d.Real lfDenom = wFilter.getWidefieldDenominator( wienParam );
	    lowFreqResult.times( lfDenom );
	    
	    //Vec2d.Cplx apoLowFreq = Vec2d.createCplx(2*w,2*h);
	    //otfPr.writeApoVector( apoLowFreq, 0.4, 1.2);
	    //lowFreqResult.times(apoLowFreq);
	    
	    if (visualFeedback>0)
		pwSt2.addImage( SimUtils.pwSpec( lowFreqResult), "filtered Widefield" );
	    spSt2.addImage( SimUtils.spatial( lowFreqResult), "filtered Widefield" );

	}	

	// stop timers
	tRec.stop();	
	tAll.stop();

	// output parameters
	Tool.trace( "\n"+param.prettyPrint(true));

	// output timings
	Tool.trace(" ---- Timings setup ---- ");
	if (findPeak)
	Tool.trace(" Parameter estimation / fit:  "+tEst);
	if (refinePhase)
	Tool.trace(" Phase refine:                "+tPha);
	Tool.trace(" Wiener filter creation:      "+tWien);
	Tool.trace(" ---- Timings reconstruction ---- ");
	Tool.trace(" Input FFTs, data from CPU:   "+tInFft);
	Tool.trace(" Band separation:             "+tBandSep);
	Tool.trace(" OTF multiplication:          "+tOtfAppl);
	Tool.trace(" Freq vector placement:       "+tFqPlace);
	Tool.trace(" Freq vector shifts (FFTs):   "+tFqShift);
	Tool.trace(" Linear algebra:              "+tLinAlg);
	Tool.trace(" Output FFT, data to CPU:     "+tCpyBack);
	Tool.trace(" Full Reconstruction:         "+tRec);
	Tool.trace(" ---");
	Tool.trace(" All:                         "+tAll);

	// DONE, display all results
	pwSt.display();
	pwSt2.display();
	spSt.display();
	spSt2.display();
	*/


}


