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

package org.fairsim.transport;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;
import java.nio.ByteOrder;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.Arrays;

import org.fairsim.linalg.Vec2d;
import org.fairsim.utils.Tool;

/** Class to encapsulate image data for network send */
public class ImageWrapper {

    final int maxWidth, maxHeight;
    
    long seqNr;
    int width, height, bpp;
    int posA, posB, pos0, pos1, pos2;

    long timeCamera=0, timeCapture=0, timeRecord=0;

    byte []	buffer;
    ByteBuffer	header;

    /** Create a new ImageWrapper, for images sized maxBytes */
    public ImageWrapper( int maxWidth, int maxHeight ) {
	this.maxWidth = maxWidth;
	this.maxHeight= maxHeight;
	buffer = new byte[ maxWidth * maxHeight*2 + 128+ 2];
	header = ByteBuffer.wrap( buffer, 0, 128);
	header.order( ByteOrder.LITTLE_ENDIAN );
    }

    /** Create an ImageWrapper from 2-byte pxl data */
    public void copy( short [] dat, int w, int h) {
	width=w; height=h; bpp=2;

	if (w*h>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, 128, buffer.length-128);
	bb.order( ByteOrder.LITTLE_ENDIAN );
	bb.asShortBuffer().put( dat, 0, w*h); 
    }

    /** Create an ImageWrapper from 2-byte pxl data */
    public void copyCrop( short [] dat, int w, int h, int origW, int origH) {
	width=w; height=h; bpp=2;

	if (origW*origH>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, 128, buffer.length-128);
	bb.order( ByteOrder.LITTLE_ENDIAN );
	ShortBuffer sb = bb.asShortBuffer();
    
	for (int y=0; y<h; y++) {
	    sb.put( dat, y*origW, w); 
	}
    }

    /** Create an ImageWrapper from 2-byte pxl data, mirroring the input in x */
    public void copyMirrorX( short [] dat, int w, int h) {
	width=w; height=h; bpp=2;

	if (w*h>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, 128, buffer.length-128);
	bb.order( ByteOrder.LITTLE_ENDIAN );
	ShortBuffer sb = bb.asShortBuffer();

	for (int y=0; y<h; y++) {
	    for (int x=0; x<w; x++) {
		//sb.put( dat, y*origW, w); 
		sb.put( dat[ y*w + (w-x-1) ] );
	    }
	}
    }

    /** Create an ImageWrapper from 2-byte pxl data, mirroring the input in x */
    public void copyCropMirrorX( short [] dat, int w, int h, int origW, int origH) {
	width=w; height=h; bpp=2;

	if (w*h>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, 128, buffer.length-128);
	bb.order( ByteOrder.LITTLE_ENDIAN );
	ShortBuffer sb = bb.asShortBuffer();

	for (int y=0; y<h; y++) {
	    for (int x=0; x<w; x++) {
		//sb.put( dat, y*origW, w); 
		sb.put( dat[ y*origW + (w-x-1) ] );
	    }
	}
    }





    /** Create an ImageWrapper from 1-byte pxl data */
    public void copy( byte [] dat, int w, int h) {
	width=w; height=h; bpp=1;

	if (w*h>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	System.arraycopy( dat, 0, buffer, 128, w*h);
    }


    /** Obtain a reference to the internal buffer (for sending) */
    public byte [] refBuffer(long sqNr) {
	seqNr=sqNr;
	writeHeader();
	return buffer;
    }

    /** Obtain a reference to the internal buffer (w/o new header, for streaming to disk) */
    public byte [] refBuffer() {
	return buffer;
    }	
    
    /** Query how many bytes of the buffer to send */
    public int bytesToSend() {
	return width*height*bpp+128;
    }


    /** read the header from an input stream */
    public int readHeader(InputStream inStr) throws IOException {
	int plen=0;
	while (plen<128) {
	    int i=inStr.read( buffer, plen, 128-plen);
	    if (i<0) return -1;	// EOF, Stream closed
	    plen+=i;
	}

	parseHeader();
	//System.out.println("w "+width+" h "+height+" bpp "+bpp);
	return width*height*bpp;
    }

    /** read the data from an input stream */
    public int readData( InputStream inStr ) throws IOException {
	int plen=0, fullLen = width*height*bpp;
	while ( plen < fullLen ) {
	    int i=inStr.read( buffer, plen+128, fullLen-plen);
	    if (i<0) return -1;	// EOF, Stream closed
	    plen+=i;
	}
	return plen;
    }

    /** write the data to an Output stream */
    public void writeData( OutputStream out) {
	// prepares the header
	writeHeader();
	try {
	    out.write( buffer, 0, bytesToSend());
	} catch (Exception e) {
	    throw new RuntimeException(e);
	}
    }


    
    /** Return A COPY of the input data. For high(er) performace,
     *  use writeToVector instead. */
    public short [] getPixels() {
	short [] ret = new short[ width*height ];
	
	if ( bpp == 2) {
	    ByteBuffer bb = ByteBuffer.wrap( buffer, 128, buffer.length-128);
	    bb.order( ByteOrder.LITTLE_ENDIAN );
	    bb.asShortBuffer().get( ret, 0, width*height); 
	} else {
	    for (int i=0; i< width*height; i++)
		ret[i] = buffer[i+128];
	}
	return ret;
    }
   
    /** Write the image data into a provided vector */
    public void writeToVector( Vec2d.Real vec ) {

	if (vec.vectorWidth() != width || vec.vectorHeight() != height )
	    throw new RuntimeException("vector to image size mismatch");

	float [] dat = vec.vectorData();

	if ( bpp == 2 ) {
	    
	    ByteBuffer bb = ByteBuffer.wrap( buffer, 128, width*height*2);
	    bb.order( ByteOrder.LITTLE_ENDIAN );
	    ShortBuffer sb = bb.asShortBuffer();
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++) {
		float val = sb.get(x+width*y);
		val = (val>0)?(val):(val+65536);
		dat[ y * width + x ] = val;
		//int pos = x + width * y;
		//byte lb = buffer[ pos*2 + 0 + 128 ];
		//byte hb = buffer[ pos*2 + 1 + 128 ];
		//if ( lb < 0 ) lb+= 256;
		//if ( hb < 0 ) hb+= 256;
		//dat[pos] = hb*256+lb;

	    }
	}
	if ( bpp == 1 ) {
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++)
		dat[ y * width + x ] = buffer[ y * width + x + 128 ]; 
	}
   
	vec.syncBuffer();
    }

    /** Write the image data into a provided vector */
    public void writeToVector( Vec2d.Cplx vec ) {

	if (vec.vectorWidth() != width || vec.vectorHeight() != height )
	    throw new RuntimeException("vector to image size mismatch");

	float [] dat = vec.vectorData();

	if ( bpp == 2 ) {
	    ShortBuffer sb = ByteBuffer.wrap( buffer, 128, width*height*2).asShortBuffer();
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++) {
		dat[ (2 * y * width + x) + 0 ] = sb.get(x+y*width); 
		dat[ (2 * y * width + x) + 1 ] = 0;
	    }
	}
	if ( bpp == 1 ) {
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++) {
		dat[ (2 * y * width + x) + 0 ] = buffer[ y * width + x + 128 ]; 
		dat[ (2 * y * width + x) + 1 ] = 0; 
	    }
	}
   
	vec.syncBuffer();
    }


    /** Create a wrapped image. Convenience method. */
    public static ImageWrapper copyImage( short [] pxl, int w, int h,
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( w, h );
	ret.copy( pxl,w,h );
	ret.setPosAB(  posA, posB );
	ret.setPos012( pos0, pos1, pos2 );
	return ret;
    }

    /** Create a wrapped image. Convenience method. */
    public static ImageWrapper copyImageMirrorX( short [] pxl, int w, int h,
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( w, h );
	ret.copyMirrorX( pxl,w,h );
	ret.setPosAB(  posA, posB );
	ret.setPos012( pos0, pos1, pos2 );
	return ret;
    }

    /** Create a wrapped, cropped image. Convenience method. */
    public static ImageWrapper copyImageCrop( short [] pxl, int w, int h, int origW, int origH, 
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( w, h );
	ret.copyCrop( pxl, w,h, origW, origH );
	ret.setPosAB(  posA, posB );
	ret.setPos012( pos0, pos1, pos2 );
	return ret;
    }

    /** Create a wrapped, cropped image. Convenience method. */
    public static ImageWrapper copyImageCropMirrorX( short [] pxl, int w, int h, int origW, int origH, 
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( w, h );
	ret.copyCropMirrorX( pxl, w,h, origW, origH );
	ret.setPosAB(  posA, posB );
	ret.setPos012( pos0, pos1, pos2 );
	return ret;
    }

    void writeHeader() {
	Arrays.fill( buffer, 0, 128, (byte)0);
	
	header.putLong(   0, seqNr );
	header.putInt(   12, width*height );
	header.put(	 16, (byte)1 );	    // protocol version
	header.put(	 17, (byte)bpp);    // bytes per pxl

	header.putShort( 32, (short)width  );
	header.putShort( 34, (short)height );
	
	header.putShort( 36, (short)posA );
	header.putShort( 38, (short)posB );
	
	header.putShort( 40, (short)pos0 );
	header.putShort( 42, (short)pos1 );
	header.putInt(   44, ( int )pos2 );
    
	header.putLong( 48, timeCamera  );
	header.putLong( 56, timeCapture );
	header.putLong( 64, timeRecord  );

    }

    void parseHeader() throws BrokenHeaderException {
	seqNr	= header.getLong(0);

	int len	= header.getInt( 12 );
	int vers= header.get( 16 );
	bpp	= header.get( 17 );

	width	= header.getShort( 32 );
	height	= header.getShort( 34 );

	posA	= header.getShort( 36 );
	posB	= header.getShort( 38 );

	pos0	= header.getShort( 40 );
	pos1	= header.getShort( 42 );
	pos2	= header.getInt(   44 );

	timeCamera   = header.getLong( 48 );
	timeCapture  = header.getLong( 56 );
	timeRecord   = header.getLong( 64 );

	if ( width > maxWidth || height > maxHeight )
	    throw new BrokenHeaderException(String.format(
		"Image too large, is/max, w: %d,%d, h: %d,%d", 
		width, maxWidth, height, maxHeight));

	if ( len != width*height )
	    throw new BrokenHeaderException("Width x Height != Len");
	if ( bpp != 1 && bpp != 2 )
	    throw new BrokenHeaderException("bpp neither 1 nor 2");
	if (vers!=1)
	    throw new BrokenHeaderException("unsupported version");

    }

    public class BrokenHeaderException extends IOException {
	public BrokenHeaderException(String m) {
	    super(m);
	}
    }


    public void setPosAB( int a, int b ) {
	if (a<0 || a>=32767 || b<0 || b>=32767)
	    throw new RuntimeException("Index out of bound");
	posA = a; posB = b;
    }

    public void setPos012( int i0, int i1, int i2 ) {
	if (i0<0 || i0>=32767 || i1<0 || i1>=32767 || i2<0 || i2>=2147483647)
	    throw new RuntimeException("Index out of bound");
	pos0 = i0; pos1 = i1; pos2 = i2;
    }


    public void setTimeCamera( long val ) {
	timeCamera = val;
    }

    public void setTimeCapture( long val ) {
	timeCapture = val;
    }

    public void setTimeRecord( long val ) {
	timeRecord = val;
    }

    public int width()  { return width;  };
    public int height() { return height; };

    public int posA() { return posA; }
    public int posB() { return posB; }
    public int pos0() { return pos0; }
    public int pos1() { return pos1; }
    public int pos2() { return pos2; }

    public long timeCamera() { return timeCamera; }
    public long timeCapture() { return timeCapture; }
    public long timeRecord() { return timeRecord; }


    // just a quick test
    public static void main( String [] arg ) {

	short [] tmp  = new short[520*520];
	
	Tool.Timer t1 = Tool.getTimer();

	ImageWrapper iw =null;
	
	for (int i=0; i<1000; i++ ){
	    iw = ImageWrapper.copyImageCropMirrorX( tmp, 512, 512, 520,520, 0,0,0,0, 1 );
	}

	t1.stop();
	System.out.println( "w "+ iw.width() + " h "+iw.height()+" "+t1);

    }

}

