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
import java.util.Collection;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.fairsim.linalg.Vec2d;
import org.fairsim.utils.Base64;
import org.fairsim.utils.Tool;

/** Class to encapsulate image data for network send */
public class ImageWrapper implements Comparable<ImageWrapper> {

    private final int maxWidth, maxHeight;
    public static final int HEADERSIZE = 128, WIDTHPOSITION = 32, HEIGHTPOSITION = 34;
    private long seqNr;
    private int width, height, bpp;
    private int posA, posB, pos0, pos1, pos2;

    private long timeCamera=0, timeCapture=0, timeRecord=0;

    private byte []	buffer;
    private ByteBuffer	header;
    
    /** Create a new ImageWrapper, for images sized maxBytes */
    public ImageWrapper( int maxWidth, int maxHeight ) {
	this.maxWidth = maxWidth;
	this.maxHeight= maxHeight;
	buffer = new byte[ maxWidth * maxHeight*2 + HEADERSIZE+ 2];
	header = ByteBuffer.wrap( buffer, 0, HEADERSIZE);
	header.order( ByteOrder.LITTLE_ENDIAN );
    }
    
    ImageWrapper(ByteBuffer header) throws BrokenHeaderException {
        this(header.getShort(WIDTHPOSITION), header.getShort(HEIGHTPOSITION));
        //System.out.println(header.getLong(HEIGHTPOSITION));
        if (header.capacity() != HEADERSIZE) throw new BrokenHeaderException("Headersize missmatch: " + header.capacity() + " " + HEADERSIZE);
        byte b = 5;
        this.header.put(header.array());
        parseHeader();
    }
    
    static ByteBuffer readImageWrapperHeader(InputStream is) throws IOException {
        ByteBuffer iwHeader = ByteBuffer.wrap(new byte[HEADERSIZE]);
        iwHeader.order(ByteOrder.LITTLE_ENDIAN);
        int plen=0;
	while (plen<HEADERSIZE) {
	    int i=is.read( iwHeader.array(), plen, HEADERSIZE-plen);
	    if (i<0) throw new IOException("Error while reading LiveStackHeader: " + i);
	    plen+=i;
	}
        return iwHeader;
    }
    
    public void setSeqNr(long seqNr) {
        this.seqNr = seqNr;
    }

    /** Create an ImageWrapper from 2-byte pxl data */
    public void copy( short [] dat, int w, int h) {
	width=w; height=h; bpp=2;

	if (w*h>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, HEADERSIZE, buffer.length-HEADERSIZE);
	bb.order( ByteOrder.LITTLE_ENDIAN );
	bb.asShortBuffer().put( dat, 0, w*h); 
    }

    /** Create an ImageWrapper from 2-byte pxl data */
    public void copyCrop( short [] dat, int w, int h, int origW, int origH) {
	copyCrop(dat, w, h, origW, origH, 0, 0);
    }
    
    public void copyCrop( short [] dat, int w, int h, int origW, int origH, int offsetX, int offsetY) {
	width=w; height=h; bpp=2;
        
        if (offsetX<0 || offsetY<0)
            throw new RuntimeException("Offset is negative");
	if (origW*origH>dat.length)
	    throw new RuntimeException("Input data too short");
	if ((w+offsetX)>maxWidth || (h+offsetY)>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, HEADERSIZE, buffer.length-HEADERSIZE);
	bb.order( ByteOrder.LITTLE_ENDIAN );
	ShortBuffer sb = bb.asShortBuffer();
    
	for (int y=offsetY; y<h+offsetY; y++) {
	    sb.put( dat, (y*origW)+offsetX, w); 
	}
    }

    /** Create an ImageWrapper from 2-byte pxl data, mirroring the input in x */
    public void copyMirrorX( short [] dat, int w, int h) {
	width=w; height=h; bpp=2;

	if (w*h>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, HEADERSIZE, buffer.length-HEADERSIZE);
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
	copyCropMirrorX(dat, w, h, origW, origH, 0, 0);
    }

    public void copyCropMirrorX( short [] dat, int w, int h, int origW, int origH, int offsetX, int offsetY) {
	width=w; height=h; bpp=2;

        if (offsetX<0 || offsetY<0)
            throw new RuntimeException("Offset is negative");
	if (w*h>dat.length)
	    throw new RuntimeException("Input data too short");
	if (w>maxWidth || h>maxHeight)
	    throw new RuntimeException("Input image larger than maximum size");

	ByteBuffer bb = ByteBuffer.wrap( buffer, HEADERSIZE, buffer.length-HEADERSIZE);
	bb.order( ByteOrder.LITTLE_ENDIAN );
	ShortBuffer sb = bb.asShortBuffer();

	for (int y=offsetY; y<h+offsetY; y++) {
	    for (int x=-offsetX; x<w-offsetX; x++) {
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

	System.arraycopy( dat, 0, buffer, HEADERSIZE, w*h);
    }

    
    /** Obtain a reference to the internal buffer (for sending) */
    /*
    public byte [] refBuffer(long sqNr) {
	seqNr=sqNr;
	writeHeader();
	return buffer;
    }
    */
    
    /** Obtain a reference to the internal buffer (w/o new header, for streaming to disk) */
    public byte [] refBuffer() {
	return buffer;
    }	
    
    /** Query how many bytes of the buffer to send */
    public int bytesToSend() {
	return width*height*bpp+HEADERSIZE;
    }

    /** read the header from an input stream */
    public int readHeader(InputStream inStr) throws IOException {
	int plen=0;
	while (plen<HEADERSIZE) {
	    int i=inStr.read( buffer, plen, HEADERSIZE-plen);
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
	    int i=inStr.read( buffer, plen+HEADERSIZE, fullLen-plen);
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
	    ByteBuffer bb = ByteBuffer.wrap( buffer, HEADERSIZE, buffer.length-HEADERSIZE);
	    bb.order( ByteOrder.LITTLE_ENDIAN );
	    bb.asShortBuffer().get( ret, 0, width*height); 
	} else {
	    for (int i=0; i< width*height; i++)
		ret[i] = buffer[i+HEADERSIZE];
	}
	return ret;
    }
   
    /** Write the image data into a provided vector */
    public void writeToVector( Vec2d.Real vec ) {

	if (vec.vectorWidth() != width || vec.vectorHeight() != height )
	    throw new RuntimeException("vector to image size mismatch");

	float [] dat = vec.vectorData();

	if ( bpp == 2 ) {
	    
	    ByteBuffer bb = ByteBuffer.wrap( buffer, HEADERSIZE, width*height*2);
	    bb.order( ByteOrder.LITTLE_ENDIAN );
	    ShortBuffer sb = bb.asShortBuffer();
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++) {
		float val = sb.get(x+width*y);
		val = (val>0)?(val):(val+65536);
		dat[ y * width + x ] = val;
		//int pos = x + width * y;
		//byte lb = buffer[ pos*2 + 0 + HEADERSIZE ];
		//byte hb = buffer[ pos*2 + 1 + HEADERSIZE ];
		//if ( lb < 0 ) lb+= 256;
		//if ( hb < 0 ) hb+= 256;
		//dat[pos] = hb*256+lb;

	    }
	}
	if ( bpp == 1 ) {
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++)
		dat[ y * width + x ] = buffer[ y * width + x + HEADERSIZE ]; 
	}
   
	vec.syncBuffer();
    }

    /** Write the image data into a provided vector */
    public void writeToVector( Vec2d.Cplx vec ) {

	if (vec.vectorWidth() != width || vec.vectorHeight() != height )
	    throw new RuntimeException("vector to image size mismatch");

	float [] dat = vec.vectorData();

	if ( bpp == 2 ) {
	    ShortBuffer sb = ByteBuffer.wrap( buffer, HEADERSIZE, width*height*2).asShortBuffer();
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++) {
		dat[ (2 * y * width + x) + 0 ] = sb.get(x+y*width); 
		dat[ (2 * y * width + x) + 1 ] = 0;
	    }
	}
	if ( bpp == 1 ) {
	    for (int y=0; y<height; y++)
	    for (int x=0; x<width; x++) {
		dat[ (2 * y * width + x) + 0 ] = buffer[ y * width + x + HEADERSIZE ]; 
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
    
    public static ImageWrapper copyImageCropCentered( short [] pxl, int w, int h, int origW, int origH, 
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( origW, origH );
	ret.copyCrop( pxl, w,h, origW, origH, (origW - w) / 2, (origH - h) / 2 );
	ret.setPosAB(  posA, posB );
	ret.setPos012( pos0, pos1, pos2 );
	return ret;
    }

    /** Create a wrapped, cropped image. Convenience method. */
    public static ImageWrapper copyImageCropMirrorX( short [] pxl, int w, int h, int origW, int origH, 
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( origW, origH );
	ret.copyCropMirrorX( pxl, w,h, origW, origH );
	ret.setPosAB(  posA, posB );
	ret.setPos012( pos0, pos1, pos2 );
	return ret;
    }
    
    public static ImageWrapper copyImageCropMirrorXCentered( short [] pxl, int w, int h, int origW, int origH, 
	int posA, int posB, int pos0, int pos1, int pos2 ) {

	ImageWrapper ret = new ImageWrapper( w, h );
	ret.copyCropMirrorX( pxl, w,h, origW, origH, (origW - w) / 2, (origH - h) / 2 );
	ret.setPosAB(  posA, posB );
	ret.setPos012( pos0, pos1, pos2 );
	return ret;
    }
    
    String encodeHeader() {
        byte[] iwHeader = new byte[128];
        System.arraycopy(refBuffer(), 0, iwHeader, 0, HEADERSIZE);
        return Base64.encode(iwHeader);
    }
    
    static ByteBuffer decodeHeader(String header) {
        byte[] headerBytes = Base64.decode(header);
        ByteBuffer iwHeader = ByteBuffer.wrap(new byte[HEADERSIZE]);
        iwHeader.order(ByteOrder.LITTLE_ENDIAN);
        iwHeader.put(headerBytes);
        return iwHeader;
    }

    void writeHeader() {
	Arrays.fill( buffer, 0, HEADERSIZE, (byte)0);
	
	header.putLong(   0, seqNr );
	header.putInt(   12, width*height );
	header.put(	 16, (byte)1 );	    // protocol version
	header.put(	 17, (byte)bpp);    // bytes per pxl

	header.putShort( WIDTHPOSITION, (short)width  );
	header.putShort( HEIGHTPOSITION, (short)height );
	
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

	width	= header.getShort( WIDTHPOSITION );
	height	= header.getShort( HEIGHTPOSITION );

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

    public long seqNr() { return seqNr; }

    @Override
    public int compareTo(ImageWrapper iw) {
        int resChannel = pos1 - iw.pos1;
        if (resChannel != 0) return resChannel;
        else return (int) (seqNr - iw.seqNr);
    }

    static final class Sorter {

        private long refSeqNr;
        private final PriorityBlockingQueue<ImageWrapper> sortBuffer;
        public final int capacity = 20;
        
        public Sorter(int capacity) {
            sortBuffer = new PriorityBlockingQueue<>(capacity);
            resetRefSeqNr();
        }
        
        public Sorter() {
            sortBuffer = new PriorityBlockingQueue<>(capacity);
            resetRefSeqNr();
        }

        public ImageWrapper poll() throws SorterException {
            if (refSeqNr == Long.MAX_VALUE) {
                for (ImageWrapper iw : sortBuffer) {
                    if (iw.seqNr < refSeqNr) refSeqNr = iw.seqNr;
                }
                if (refSeqNr == Long.MAX_VALUE) throw new RuntimeException("this should never happen");
            }
            ImageWrapper iw = sortBuffer.poll();
            if (iw == null) return iw;
            if (iw.seqNr == refSeqNr) {
                refSeqNr++;
                return iw;
            } else {
                sortBuffer.put(iw);
                throw new SorterException("Wrong order " + iw.seqNr + " " + refSeqNr);
            }
        }
        
        public void add(ImageWrapper iw) throws SorterException {
            if (!isFull()) sortBuffer.put(iw);
            else throw new SorterException("Buffer overflow " + sortBuffer.size());
        }
        
        public int size() {
            return sortBuffer.size();
        }
        
        public boolean isFull() {
            return sortBuffer.size() >= capacity;
        }
        
        public boolean isEmpty() {
            return sortBuffer.isEmpty();
        }
        
        public void resetRefSeqNr() {
            refSeqNr = Long.MAX_VALUE;
        }
        
        public void clear() {
            sortBuffer.clear();
        }
        
        public class SorterException extends Exception {
            SorterException(String message) {
                super(message);
            }
        }
    }
    
    // just a quick test
    public static void main(String[] arg) throws Sorter.SorterException, InterruptedException {

        Sorter sorter = new Sorter(4);
        
        for (int i = 3; i > 0; i--) {
            System.out.println(i);
            ImageWrapper iw = new ImageWrapper(512, 512);
            iw.seqNr = i;
            sorter.add(iw);
        }
        
        System.out.println(4);
        ImageWrapper iw = new ImageWrapper(512, 512);
        iw.seqNr = 4;
        sorter.add(iw);
        
        
        for (int i = 0; i < 4; i++) {
            System.out.print(10*i + " : ");
            ImageWrapper iww = sorter.poll();
            System.out.println(iww.seqNr);
        }
        
        

    }


}

