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

import org.fairsim.linalg.AbstractVectorReal;
import org.fairsim.linalg.Vec;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.Vec3d;

import java.nio.ByteBuffer;


/** Vectors in native C code through JNI */
class AccelVectorReal extends AbstractVectorReal {

    /** stores the C pointer */
    final long natData;
    boolean deviceNew = false, hostNew = false;

    public int ourCopyMode ;

    final protected AccelVectorFactory ourFactory;

    /** creates a new vector, allocates memory */
    AccelVectorReal(AccelVectorFactory vf, int n ){
	super(n);
	
	ourFactory = vf;
	ourCopyMode = vf.defaultCopyMode();
	
	natData = alloc( vf, n );
	if (natData == 0) {
	    throw new java.lang.OutOfMemoryError("No memory for"+
		"allocating native vector");
	}
    }

    /** tells the native code to deallocate its memory */
    @Override
    protected void finalize() {
	dealloc( natData );
    }

    @Override
    public AccelVectorReal duplicate() {
	AccelVectorReal ret  = ourFactory.createReal(elemCount);
	ret.copy( this );
	return ret;
    }

    @Override
    public void readyBuffer() {
	if (deviceNew) {
	    long buffer = 0;
	    if (ourCopyMode == 2) {
		buffer = ourFactory.getNativeHostBuffer();
		if (buffer==0)
		    throw new RuntimeException("null pointer (caught in JAVA)");
	    }
	    copyBuffer( natData, this.data, buffer, true, ourCopyMode );
	}
	deviceNew = false;
    };
    
    @Override
    public void syncBuffer() {
	long buffer = 0;
	if (ourCopyMode == 2) {
            buffer = ourFactory.getNativeHostBuffer();
	    if (buffer==0)
		throw new RuntimeException("null pointer (caught in JAVA)");
	}
	copyBuffer( natData, this.data, buffer,  false, ourCopyMode );	
	hostNew = false;
    };

    public void readyDevice() {
	if ( hostNew )
	    syncBuffer();
    }

    void freeHostCopyBuffer(int i) {
	System.out.println("Buffer freed: " +i);
    }


    @Override
    public void makeCoherent() {
	if ( hostNew && deviceNew ) 
	    throw new RuntimeException("Changes occured to both device and host memory");
	if (hostNew)
	    syncBuffer();
	if (deviceNew)
	    readyBuffer();
    }




    // ------ mathematical functions ------

    /** see if we have to call the 'super' function */
    boolean callSuper( Vec.Real v ) {
	if (v==null) throw new NullPointerException("Input vector is null");
	if (! ( v instanceof AccelVectorReal )) 
	    return true;
	
	Vec.failSize(this, v);
	return false;
    }


    @Override
    public void add( Vec.Real ... v ) {
	if (v==null || v.length==0)
	    return;
	
	// Todo: currently if one vector is not our type,
	// everything goes through the slow function
	for (Vec.Real v1 : v)
	if (! ( v1 instanceof AccelVectorReal )) {
	    super.add(v);
	    System.out.println("using fallback");
	    return;
	}
	
	readyDevice();
	
	Vec.failSize(this, v);
	for (Vec.Real v1 : v) {
	    ((AccelVectorReal)v1).readyDevice();
	    nativeAdd( this.natData, ((AccelVectorReal)v1).natData, elemCount );
	}
	
	deviceNew = true;
    }

    @Override
    public void addConst(float a) {
	readyDevice();
	nativeADDCONST( this.natData, elemCount, a );
    }

    @Override
    public void axpy( float a, Vec.Real x ) {
	if ( callSuper(x) ) {	
	    super.axpy(a,x);
	    return;
	}
	readyDevice();
	((AccelVectorReal)x).readyDevice();
	nativeAXPY( a, this.natData, ((AccelVectorReal)x).natData, elemCount );
	deviceNew = true;
    }

    @Override 
    public void times( Vec.Real x ) {
	if ( callSuper(x) ) {	
	    super.times(x);
	    return;
	}
	readyDevice();
	((AccelVectorReal)x).readyDevice();
	nativeTIMES( this.natData, ((AccelVectorReal)x).natData, elemCount );
	deviceNew = true;
    }
    
    @Override
    public void elementwiseDivision(Vec.Real x) {
	/*
        Vec.failSize(this, x);
	this.readyBuffer();
	float [] id = x.vectorData();
	for (int i=0;i<elemCount;i++) 
	    data[i] = data[i] / id[i];
	this.syncBuffer();
        */
    }
  
    @Override
    public double norm2() {
	readyDevice();
	return nativeREDUCE( this.natData, elemCount, true );
    }


    @Override
    public void zero() {
	nativeZero( this.natData , elemCount);
	deviceNew = true;
    }
    
    @Override
    public void scal( float a ) {
	readyDevice();
	nativeScal( this.natData, elemCount, a);
	deviceNew = true;
    }

    
    // ------ direct native methods ------

    /** set the vector to zero */
    public native void nativeZero(long addr, int elem);

    /** multiply by scalar */
    public native void nativeScal(long addr, int elem, float a);

    // ------ native methods ------

    /** Allocate vector in native code */
    native long alloc(AccelVectorFactory vf, int n);
    
    /** Disallocate vector */
    native void dealloc(long adrr);

    /** sync to / from java code */
    native void copyBuffer( long addr, float [] jvec, long buffer, boolean toJava, int copyMode );
    
    /** add vectors */
    native void nativeAdd( long addrThis, long addrOther, int len );
	
    /** add constant float value */
    native void nativeADDCONST( long addrThis, int len, float a );
    
    /** axpy vectors */
    native void nativeAXPY( float a, long addrThis, long addrOther, int len );
    
    /** times vectors */
    native void nativeTIMES( long addrThis, long addrOther, int len );

    /** norm2 */
    native double nativeREDUCE( long addrThis, int len, boolean sqr );


}


class AccelVectorReal2d extends AccelVectorReal implements Vec2d.Real {

    final int width, height;

    AccelVectorReal2d(AccelVectorFactory vf, int w, int h) {
	super(vf, w*h);
	width=w; height=h;
    }

    // ------ Interface implementation ------

    @Override
    public int vectorWidth() { return width; }
    @Override
    public int vectorHeight() { return height; }

    @Override
    public void paste( Vec2d.Real in, int x, int y, boolean zero ) {
	Vec2d.paste( in, this, 0, 0, in.vectorWidth(), in.vectorHeight(), x, y, zero);
    }

    @Override
    public AccelVectorReal2d duplicate() {
	AccelVectorReal2d ret = ourFactory.createReal2D(width, height);
	ret.copy( this );
	return ret;
    }

    @Override
    public void set(int x, int y, float a ) {
	readyBuffer();
	//nativeSet(natData, x,y, width, a);
	data [ x + y*width ] = a;
	hostNew = true;
    }
    

    @Override
    public float get(int x, int y) {
	readyBuffer();
	//float  res = nativeGet( natData, x,y, width);
	//return res;
	return data [ x + y*width ];
    }

    @Override
    public void slice( Vec3d.Real in , int s ) {
	throw new RuntimeException("Currently not implemented for AccelVector");
    }
    @Override
    public void project( Vec3d.Real in , int s , int e) {
	throw new RuntimeException("Currently not implemented for AccelVector");
    }
    @Override
    public void project( Vec3d.Real in ) {
	throw new RuntimeException("Currently not implemented for AccelVector");
    }


    @Override
    public void setFrom16bitPixels( short [] in ) {
	
	if (elemCount > AccelVectorFactory.nativeBufferSize/4)
	    throw new RuntimeException("Size exceeds buffer");
	
	long ptrbufHost	= ourFactory.getNativeHostBuffer(); 
	long ptrbufDevice=ourFactory.getNativeDeviceBuffer();

	nativeCOPYSHORT( this.natData, ptrbufHost, ptrbufDevice, in, elemCount );
	deviceNew = true;
    }




    // ------ Native methods ------

    native void nativeCOPYSHORT( long ptrOut, long bufferHost, long bufferDevice, short [] in, int elem);

    native void  nativeSet( long data, int x, int y, int width, float in);
    native float nativeGet( long data, int x, int y, int width);
    
    //native void nativeFFT( long fftPlan, long data, boolean inverse );


}




