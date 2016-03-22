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

import java.util.Map;
import java.util.TreeMap;

/**
 * Provides FFTs for vector elements.
 */
public abstract class FFTProvider {

    /** key to store instances */
    private static class FFTkey implements  Comparable<FFTkey> { 
	final int d,x,y ; 
	FFTkey( int xi ) {
	    d=1; x=xi; y=-1;
	}
	FFTkey( int xi, int yi ) { 
	    d=2; x=xi; y=yi;
	}
	@Override
	public int compareTo(FFTkey t) {
	    if (d != t.d) return (t.d -d );
	    if (x != t.x) return (t.x -x );
	    if (y != t.y) return (t.y -y );
	    return 0;
	}
    }
    
    /** FFT instances */
    static private Map<FFTkey, Long> instances; 
    
    /** static initialization of the instances list. */
    static {
	if (instances==null) {
	    instances = new TreeMap<FFTkey, Long>();
	}
    }

    /** returns an instance, creates one if none exists */
    static long getOrCreateInstance(int x, int y) {
	FFTkey k = new FFTkey( x,y);
	
	Long ffti = instances.get(k);
	if (ffti!=null) return ffti;
	
	ffti = nativeCreatePlan2d( k.x, k.y );
	
	instances.put( k , ffti );
	return ffti;
    }

    native static long nativeCreatePlan2d( int w, int h);

}
