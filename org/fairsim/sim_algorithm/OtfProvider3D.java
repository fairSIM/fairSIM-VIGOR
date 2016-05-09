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

package org.fairsim.sim_algorithm;

import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.Vec3d;
import org.fairsim.linalg.Cplx;
import org.fairsim.linalg.MTool;

import org.fairsim.utils.Tool;
import org.fairsim.utils.Conf;
import org.fairsim.utils.SimpleMT;


/**
 * OTFs and associated functions (attenuation, apotization, ...).
 * Provides loading, saving, simple estimation, conversion
 * from 2D (radially symmetric, in phys. units) to 3D vectors.
 * */
public class OtfProvider3D {
    
    // --- internal parameters ----

    // vals[band][idx], where idx = cycles / cyclesPerMicron
    private Vec2d.Cplx [] vals	=  null; 
    
    // physical units
    private double cyclesPerMicronLateral;
    private double cyclesPerMicronAxial;
    private double na, lambda, cutOff;

    // vector size
    private int samplesLateral, samplesAxial;
    private boolean isMultiBand=true;
    private int maxBand=3;
    
    private double vecCyclesPerMicronLateral=-1;
    private double vecCyclesPerMicronAxial=-1;


    // ------ setup / access methods -------

   /** Returns a short description (GUI display, etc) */
    public String printState(boolean html) {
	String ret ="";
	if (html)
	    ret = String.format("NA %4.2f, \u03bb<sub>em</sub> %4.0f, ", na, lambda);
	else 
	    ret = String.format("NA %4.2f, lambda %4.0f, ", na, lambda);
	
	ret+=String.format("(from file)");
	return ret;
    }


    /** Return the OTF cutoff, unit is cycles/micron */
    public double getCutoff() {
	return cutOff;
    }

    /** Get the OTF value at 'cycl'.  
     *  @param band OTF band
     *  @param xycycl xy-lateral Position in cycles/micron
     *  @param zcycl z-axial Position in cycles/micron
     *  @param att  If true, return attenuated value (see {@link #setAttenuation})
     * */
    public Cplx.Float getOtfVal(int band, double xycycl, double zcycl) {
	// checks
	if ( !this.isMultiBand ) 
	    band=0;
	if ((band >= maxBand)||(band <0))
	    throw new IndexOutOfBoundsException("band idx too high or <0");
	if (( xycycl < 0 ) || (zcycl < 0))
	    throw new IndexOutOfBoundsException("cylc negative!");
	
	// out of support, return 0
	if (( xycycl >= cutOff )||(zcycl >= cutOff))
	    return Cplx.Float.zero();
	
	final double xpos = xycycl / cyclesPerMicronLateral;
	final double zpos = zcycl  / cyclesPerMicronAxial;
	
	if ( Math.ceil(xpos) >= samplesLateral || Math.ceil(zpos) >= samplesAxial )
	    return Cplx.Float.zero();

    
	// for now, linear interpolation, could be better with a nice cspline
	int lxPos = (int)Math.floor( xpos );	
	int hxPos = (int)Math.ceil(  xpos );
	float fx = (float)(xpos - lxPos);
	
	int lzPos = (int)Math.floor( zpos );	
	int hzPos = (int)Math.ceil( zpos );
	float fz = (float)(zpos - lzPos);
    
	Cplx.Float r1 = vals[band].get(lxPos, lzPos).mult( 1 - fx );
	Cplx.Float r2 = vals[band].get(hxPos, lzPos).mult( fx );
	Cplx.Float r3 = vals[band].get(lxPos, hzPos).mult( 1 - fx );
	Cplx.Float r4 = vals[band].get(hxPos, hzPos).mult( fx );
    
	Cplx.Float r5 = Cplx.add( r1, r2 ).mult( 1 - fz );
	Cplx.Float r6 = Cplx.add( r3, r4 ).mult( fz );
	
	return Cplx.add( r5, r6 );

    }
   
    /** Sets pixel size, for output to vectors
     *	@param cyclesPerMicron Pixel size of output vector, in cycles/micron */
    public void setPixelSize( double cyclesPerMicronLateral, double cyclesPerMicronAxial ) {
	if (cyclesPerMicronLateral<=0 || cyclesPerMicronAxial <= 0)
	    throw new IllegalArgumentException("pxl size must be positive");
	vecCyclesPerMicronLateral=cyclesPerMicronLateral;
	vecCyclesPerMicronAxial  =cyclesPerMicronAxial;
    }

    // ------ Attenuatson -----
    // ------ applying OTF to vectors -------

    /** Multiplies / outputs OTF to a vector. Quite general function,
     *  some wrappers are provided for conveniece. 
     *  @param vec  Vector to write / multiply to
     *	@param band OTF band 
     *	@param kx OTF center position offset x
     *	@param ky OTF center position offset y
     *  @param useAtt if to use attenuation (independent of how {@link #switchAttenuation} is set)
     *  @param write  if set, vector is overridden instead of multiplied
     *  */
    public void otfToVector( final Vec3d.Cplx vec, final int band, 
	final double kx, final double ky, final boolean write ) {
	
	// parameters
	if (vecCyclesPerMicronLateral <=0 || vecCyclesPerMicronAxial <=0 )
	    throw new IllegalStateException("Vector pixel size not initialized");
	final int w = vec.vectorWidth(), h = vec.vectorHeight(), d = vec.vectorDepth();

	// loop output vector
	new SimpleMT.StrPFor(0,d) {
	    public void at(int z) {
		for (int y=0; y<h; y++) 
		for (int x=0; x<w; x++) {
		    // wrap to coordinates: x in [-w/2,w/2], y in [-h/2, h/2]
		    double xh = (x<w/2)?( x):(x-w);
		    double yh = (y<h/2)?(-y):(h-y);
		    double zh = (z<d/2)?( z):(d-z);
		    
		    // from these, calculate distance to kx,ky, convert to cycl/microns
		    double rad = MTool.fhypot( xh-kx, yh-ky );
		    double cycllat = rad * vecCyclesPerMicronLateral;
		    double cyclax  = zh  * vecCyclesPerMicronAxial;
		    
		    // over cutoff? just set zero
		    if ( cycllat > cutOff ) {
			vec.set(x,y,z, Cplx.Float.zero());
		    } 
		    
		    // within cutoff?
		    if ( cycllat <= cutOff ) {
		    
			// get the OTF value
			Cplx.Float val = getOtfVal(band, cycllat, cyclax);

			// multiply to vector or write to vector
			if (!write) {
			    vec.set(x, y, z, vec.get(x,y,z).mult( val.conj() ) );
			} else {
			    vec.set(x, y, z, val );
			}
		    }
		}
	    }
	}; 

    }

    // ------ Applying OTF to vectors ------
    
    /** Multiplied conjugated OTF to a vector.
     *  The desired vector pixel size has to be set (via {@link #setPixelSize}) first.
     *  @param vec  Vector to write to
     *	@param band OTF band */
    public void applyOtf(Vec3d.Cplx vec, final int band) {
	otfToVector( vec, band, 0, 0, false ) ; 
    }

    /** Multiplied conjugated OTF to a vector.
     *  The desired vector pixel size has to be set (via {@link #setPixelSize}) first.
     *  @param vec  Vector to write to
     *	@param band OTF band 
     *	@param kx OTF center position offset x
     *	@param ky OTF center position offset y */
    public void applyOtf(Vec3d.Cplx vec, final int band, double kx, double ky ) {
	otfToVector( vec, band, kx, ky, false ) ; 
    }

    /** Create a 3-dimension, radial symmetric vector of the OTF, centered at kx,ky.
     *  The desired vector pixel size has to be set (via {@link #setPixelSize}) first.
     *  @param vec  Vector to write to
     *	@param band OTF band 
     *	@param kx Position / offset kx
     *	@param ky Poistion / offset ky
     *	*/
    public void writeOtfVector(final Vec3d.Cplx vec, final int band, 
	final double kx, final double ky) {
	otfToVector( vec, band, kx, ky, true ) ; 
    }
   
    /** Copy an existing OTF */
    public OtfProvider3D duplicate() {
	
	OtfProvider3D ret = new OtfProvider3D();

	ret.cyclesPerMicronLateral	= this.cyclesPerMicronLateral;
	ret.cyclesPerMicronAxial	= this.cyclesPerMicronAxial;
	ret.na				= this.na;
	ret.lambda			= this.lambda;
	ret.cutOff			= this.cutOff;
	ret.samplesLateral		= this.samplesLateral;
	ret.samplesAxial		= this.samplesAxial;
	ret.isMultiBand			= this.isMultiBand;
	ret.maxBand			= this.maxBand;
	ret.vecCyclesPerMicronLateral	= this.vecCyclesPerMicronLateral;
	ret.vecCyclesPerMicronAxial	= this.vecCyclesPerMicronAxial;

	ret.vals = new Vec2d.Cplx[ this.vals.length ];
	for (  int i=0; i< this.vals.length; i++)
	    ret.vals[i] = this.vals[i].duplicate();

	return ret;
    }

    // ------ Load / Save operations ------

    /** Create an OTF stored in a string representation, usually read from
     *  file. 
     *	@param cfg The config to load from
     *  */
    public static OtfProvider3D loadFromConfig( Conf cfg ) 
	throws Conf.EntryNotFoundException {

	Conf.Folder fld = cfg.r().cd("otf3d");

	OtfProvider3D ret = new OtfProvider3D();

	// main parameters
	ret.na	    = fld.getDbl("NA").val();
	ret.lambda  = fld.getInt("emission").val();
	ret.cutOff  = 1000 / (ret.lambda / ret.na /2);
	
	if (!fld.contains("data"))
	    throw new RuntimeException("No data section found, needed for 3d");

	    
	// copy parameters
	Conf.Folder data = fld.cd("data");
	ret.maxBand = data.getInt("bands").val();
	ret.isMultiBand = (ret.maxBand>1);
	
	ret.samplesLateral  = data.getInt("samples-lateral").val(); 
	ret.samplesAxial    = data.getInt("samples-axial").val(); 

	ret.cyclesPerMicronLateral = data.getDbl("cycles-lateral").val();
	ret.cyclesPerMicronAxial   = data.getDbl("cycles-axial").val();
	
	// init bands
	ret.vals	= Vec2d.createArrayCplx( ret.maxBand, 
	    ret.samplesLateral, ret.samplesAxial );

	// copy bands
	for (int b=0; b<ret.maxBand; b++) {
	    byte  [] bytes = data.getData(String.format("band-%d",b)).val();
	    float [] val   = Conf.fromByte( bytes );
	  
	    if (val.length != 2*ret.samplesAxial * ret.samplesLateral )
		throw new RuntimeException("OTF read data length mismatch: "+val.length+" "+
		    ret.samplesAxial+" "+ret.samplesLateral);

	    int i=0;
	    for (int  z=0;  z< ret.samplesAxial   ;  z++)  
	    for (int xy=0; xy< ret.samplesLateral ; xy++) { 
		ret.vals[b].set(xy,z , new Cplx.Float( val[2*i], val[2*i+1]));	    
		i++;
	    }
	}

	return ret;
    
    }






    // ------ Testing ------

    /** For testing, outputs some OTF to stdout */
    public static void main( String args [] ) throws Exception {

	if (args.length==0) {
	    System.out.println("Use: io - Input/Output OTF ");
	    return;
	}

	// timing
	/*
	if (args[0].equals("t")) {

	    otf.setPixelSize(0.023, 0.133);
	    Vec3d.Cplx [] test = Vec3d.createArrayCplx(15,512,512);

	    for (int loop=0; loop<5; loop++) {

		// Test1: Apply OTF on the fly
		Tool.Timer t2 = Tool.getTimer();
		t2.start();
		for (int i=0;i<15; i++) 
		    otf.applyOtf( test[i], 0 ) ;
		t2.stop();
		
		Tool.trace("OTF on-fly: "+t2);
	    }
	} */
	

	// output
	if (args[0].equals("io")) {
	    
	    Conf cfg = Conf.loadFile( args[1] );
	    OtfProvider3D otf = OtfProvider3D.loadFromConfig( cfg ); 
	
	    System.out.println("# Size: "+otf.cyclesPerMicronLateral+" "+otf.cyclesPerMicronAxial+
		 "( "+otf.vals[0].vectorWidth()+" "+ otf.vals[0].vectorHeight()+" ) ");
	    

	    // output the vector as read in
	    for (int z=0; z<otf.vals[0].vectorHeight();z++)  {
		for (int l=0; l<otf.vals[0].vectorWidth();l++) {
			System.out.println(String.format(" %5.3f %5.3f %6.4f %6.4f %6.4f #A",
			    l*otf.cyclesPerMicronLateral, z*otf.cyclesPerMicronAxial,
			    //l*1., z*1.,
			    otf.vals[0].get(l,z).re,
			    otf.vals[1].get(l,z).re,
			    otf.vals[2].get(l,z).re ));
		}
		System.out.println("    #A");
	    }

	    for (float cyclax = 0; cyclax < 8; cyclax +=0.025 ) {
		for (float cycllat = 0; cycllat < 8; cycllat +=0.025 ) {
		    System.out.println(String.format(" %5.3f %5.3f %6.4f %6.4f %6.4f #B", 
			cycllat, cyclax, 
			otf.getOtfVal(0, cycllat, cyclax).re, 
			otf.getOtfVal(1, cycllat, cyclax).re, 
			otf.getOtfVal(2, cycllat, cyclax).re ));
		}
		System.out.println("   #B");
	    }
	}

	Tool.shutdown();
    }

}
