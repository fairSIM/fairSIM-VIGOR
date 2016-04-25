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

import org.fairsim.linalg.Vec;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.Vec3d;
import org.fairsim.linalg.VectorFactory;

import org.fairsim.utils.Tool;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;

public class AccelVectorFactory implements VectorFactory {


    // native-memory buffer for efficient copying
    static final int nativeBufferSize  = 1024*1024*8;
    static final int nativeBufferCount = 32;

    // TODO: Other thread-save collections?
    final BlockingQueue<Long> nativeDeviceBuffers = 
	new ArrayBlockingQueue<Long>(nativeBufferCount*2);
    
    final BlockingQueue<Long> nativeHostBuffers = 
	new ArrayBlockingQueue<Long>(nativeBufferCount*2);

    {
	for (int i=0; i<nativeBufferCount; i++)
	    nativeDeviceBuffers.offer( nativeAllocMemory( nativeBufferSize)); 
	for (int i=0; i<nativeBufferCount; i++)
	    nativeHostBuffers.offer( nativeAllocMemoryHost( nativeBufferSize)); 
    }
    

    /** Obtain a buffer in pinned native host memory (please return it!) */
    long getNativeHostBuffer() {
	long ret;
	try {
	    ret   = nativeHostBuffers.take();
	} catch (Exception e) {
	    throw new RuntimeException(e);
	}
	Tool.trace("handed out host buffer: " + ret);
	return ret;
    }
    
    /** Obtain a buffer in native device memory (please return it!) */
    long getNativeDeviceBuffer() {
	long ret;
	try {
	    ret   = nativeDeviceBuffers.take();
	} catch (Exception e) {
	    throw new RuntimeException(e);
	}
	return ret;
    }
    
    /** Return a native host-side buffer after use */
    void returnNativeHostBuffer( long buf ) {
	System.out.println("just got back host buffer: " + buf);
	nativeHostBuffers.offer( buf );
    }
    
    /** Return a native device-side buffer after use */
    void returnNativeDeviceBuffer( long buf ) {
	System.out.println("just got back DEVICE buffer: " + buf);
	nativeDeviceBuffers.offer( buf );
    }


    private AccelVectorFactory() {};

    static public AccelVectorFactory getFactory() {
	return new AccelVectorFactory();
    }

    public AccelVectorReal createReal( int n) {
	return new AccelVectorReal( this, n);
    }

    public AccelVectorCplx createCplx( int n) {
	return new AccelVectorCplx( this, n);
    }

    public AccelVectorReal2d createReal2D(int w, int h) {
	return new AccelVectorReal2d( this, w, h);
    }
    
    public AccelVectorCplx2d createCplx2D(int w, int h) {
	return new AccelVectorCplx2d( this, w, h);
    }
    
    public Vec3d.Real createReal3D(int w, int h, int d) {
	throw new RuntimeException("Currently not implemented for AccelVector");
    }
    
    public Vec3d.Cplx createCplx3D(int w, int h, int d) {
	throw new RuntimeException("Currently not implemented for AccelVector");
    }


    public void syncConcurrent() {
	nativeSync();
    }

    native static void nativeSync();
    native long nativeAllocMemory(int size);
    native long nativeAllocMemoryHost(int size);
}
