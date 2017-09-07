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

import java.io.File;
import java.io.FileOutputStream;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import org.fairsim.utils.Tool;

/** Provides the control interface for live mode */
public class ImageDiskWriter {

    final File saveFolder;
    final BlockingQueue<ImageWrapper> saveQueue ;
    final int saveQueueMax ;
    
    ImageSaveThread fileRunner = null;

    int fullBufferCount = 0;
    
    /** Initialize a LiveStreamWriter, saving a given folder */
    public ImageDiskWriter( String folder, int bufferSize ) {
	saveFolder = new File( folder );
	if (!saveFolder.exists() || !saveFolder.isDirectory())
	    throw new RuntimeException("save folder path is not a folder");
	saveQueue = new ArrayBlockingQueue<ImageWrapper>( bufferSize);
	saveQueueMax = bufferSize;
    }

    /** get the amount of free space */
    public long getSpace() {
	return saveFolder.getUsableSpace();
    }

    /** calcualte how many seconds of stream can be stored.
     *  @param imgSize	    Image widht, height (in pxl)
     *  @param imgChannels  Number of channels
     *  @param imgPerSec    Raw data rate, in img/sec
    */
    public double getTimeLeft(int imgSize, int imgChannels, double imgPerSec) {
	double bytesLeft = getSpace();
	double bytesPerSecond = (imgSize*imgSize*2)*imgPerSec*imgChannels;
	return bytesLeft/bytesPerSecond;
    }


    /** Queue an image to be saved */
    public void saveImage( ImageWrapper iw ) {
	// only accept images if the writer is set to record
	if (fileRunner==null) return;

	boolean isQueued = saveQueue.offer( iw );
	if (!isQueued) fullBufferCount++;
    }


    /** start (or restart to a new file) streaming data to disk */
    public void startRecording(String prefix, LiveStack.Header header) throws java.io.IOException {

	// stop current recording process (if any)
	if (fileRunner != null)
	    stopRecording();

	// generate filename
	String nowAsISO = header.timestamp;
	File outfile = new File( saveFolder , prefix+"_"+nowAsISO+".livestack");

	fileRunner = new ImageSaveThread( outfile, prefix+"_"+nowAsISO, header );
	fileRunner.start();
    }


    /** stop the recording process */
    public void stopRecording() {
	fileRunner.stopSoon = true;
	fileRunner.interrupt();
	try {
	    fileRunner.join();
	} catch (java.lang.InterruptedException e) {
	    throw new RuntimeException(e);
	}
	fileRunner = null;
    }

    /** Get how much the buffer is filled, in percent */
    public int bufferState() {
	return (100*saveQueue.size()) / saveQueueMax ;
    }

    /** Get and reset the number of dropped frames since last call */
    public int nrDroppedFrames() {
	int ret = fullBufferCount;
	fullBufferCount =0;
	return ret;
    }
    


    /** Thread streaming data to disk */
    class ImageSaveThread extends Thread {
    
	final FileOutputStream outfile ;
	final String filename;
        final LiveStack.Header lsHeader;
	boolean stopSoon = false;

	ImageSaveThread(File out, String name, LiveStack.Header header) throws java.io.IOException {
	    outfile  = new FileOutputStream(out);
	    filename = name;
            lsHeader = header;
	}


	public void run() {
	    
	    Tool.trace("-disk- Writing to "+ filename);
            try {
                lsHeader.write(outfile);
            } catch (java.io.IOException e) {
                throw new RuntimeException(e);
            }

	    while (!stopSoon) {


		// retrieve the next image to save
		ImageWrapper imgToSave = null;
		try {
		    imgToSave = saveQueue.take();
		} catch (InterruptedException e) {
		    if (stopSoon) break;
		}

		// write the full buffer to disk
		try {
		if (imgToSave!=null)
		    outfile.write( imgToSave.refBuffer(), 0, imgToSave.bytesToSend());
		    outfile.flush();
		} catch ( java.io.IOException e ) {
		    throw new RuntimeException(e);
		}
	    }
	    // close the file, exit thread
	    try {
		outfile.close();
	    } catch (java.io.IOException e) {
		throw new RuntimeException(e);
	    }
	    Tool.trace("-disk- Stopped writing to "+filename);
	}

    }




}

