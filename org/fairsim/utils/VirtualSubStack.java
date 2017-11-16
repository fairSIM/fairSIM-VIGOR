package org.fairsim.utils;

import ij.process.ImageProcessor;
import ij.ImageStack;

import java.util.List;

/** This class represents an array of disk-resident images. */
public class VirtualSubStack extends ImageStack {
	
	final ImageStack origStack;
	final List<Integer> framePos;


	/** Creates a new view on a stack */
	public VirtualSubStack( ImageStack origStack, List<Integer> framePos ) {
	    // TODO: sanity checks (sizes)!
	    super( origStack.getWidth(), origStack.getHeight(), null);
	    this.origStack = origStack;
	    this.framePos  = framePos;
	}

	    
	/** Does nothing. */
	public void addSlice(String name) {
	}

	/** Does nothing. */
	public void addSlice(String sliceLabel, Object pixels) {
	}

	/** Does nothing.. */
	public void addSlice(String sliceLabel, ImageProcessor ip) {
	}
	
	/** Does noting. */
	public void addSlice(String sliceLabel, ImageProcessor ip, int n) {
	}

	/** Does nothing. */
	public void deleteSlice(int n) {
	}

	/** Does nothing. */
	public void deleteLastSlice() {
	}
	   
	/** Returns the pixel array for the specified slice, were 1<=n<=nslices. */
	public Object getPixels(int n) {
		ImageProcessor ip = getProcessor(n);
		if (ip!=null)
			return ip.getPixels();
		else
			return null;
	}		
	
	/** Does nothing. */
	public void setPixels(Object pixels, int n) {
	}

	
	/** Returns a processor from the original stack at the position
	 * defined in framePos */
	public ImageProcessor getProcessor(int n) {
	    return origStack.getProcessor( framePos.get(n-1) );
	}
 
	/** Does nothing, fails */
	public int saveChanges(int n) {
		return -1;
	}

	 /** Returns the number of slices in this stack. */
	public int getSize() {
		return framePos.size();
	}

	/** Returns the label of the Nth image. */
	public String getSliceLabel(int n) {
	    //return origStack.getSliceLabel( framePos.get(n) +1 );
	    return ("OrigImageNr: "+framePos.get(n-1));
	}
	
	/** Returns null. */
	public Object[] getImageArray() {
		return null;
	}

        /** Does nothing. */
	public void setSliceLabel(String label, int n) {
	}

	/** Always return true. */
	public boolean isVirtual() {
		return true;
	}

	/** Does nothing. */
	public void trim() {
	}
	
	/** Returns the path to the directory containing the images. */
	/*
	 * public String getDirectory() {
		return path;
	} */
		
	/** Returns the file name of the specified slice, were 1<=n<=nslices. */
	/*
	public String getFileName(int n) {
		return names[n-1];
	} */
	
	/** Does nothing */
	public void setBitDepth(int bitDepth) {
	}

	/** Returns the bit depth (8, 16, 24 or 32), or 0 if the bit depth is not known. */
	public int getBitDepth() {
		return origStack.getBitDepth();
	}
	
	/*
	public ImageStack sortDicom(String[] strings, String[] info, int maxDigits) {
		int n = getSize();
		String[] names2 = new String[n];
		for (int i=0; i<n; i++)
			names2[i] = names[i];
		for (int i=0; i<n; i++) {
			int slice = (int)Tools.parseDouble(strings[i].substring(strings[i].length()-maxDigits), 0.0);
			if (slice==0) return null;
			names[i] = names2[slice-1];
			labels[i] = info[slice-1];
		}
		return this;
	} */

} 

