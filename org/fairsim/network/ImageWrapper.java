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

package org.fairsim.network;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;
import java.nio.ByteOrder;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.Arrays;

/** Class to encapsulate image data for network send */
public class ImageWrapper {

    final int maxWidth, maxHeight;
    
    long seqNr;
    int width, height, bpp;
    int posA, posB, pos0, pos1, pos2;

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


    
    /** Return A COPY of the input data */
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
    

    /** Create a wrapped image. Convenience method. */
    public static ImageWrapper copyImage( short [] pxl, int w, int h,
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( w, h );
	ret.copy( pxl,w,h );
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

    public int width()  { return width;  };
    public int height() { return height; };

    public int posA() { return posA; }
    public int posB() { return posB; }
    public int pos0() { return pos0; }
    public int pos1() { return pos1; }
    public int pos2() { return pos2; }

}

