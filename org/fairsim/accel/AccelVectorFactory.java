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

public class AccelVectorFactory implements VectorFactory {

    private AccelVectorFactory() {};

    static public VectorFactory getFactory() {
	return new AccelVectorFactory();
    }


    public Vec.Real createReal( int n) {
	return new AccelVectorReal( n);
    }

    public Vec.Cplx createCplx( int n) {
	return new AccelVectorCplx( n);
    }

    public Vec2d.Real createReal2D(int w, int h) {
	return new AccelVectorReal2d( w, h);
    }
    
    public Vec2d.Cplx createCplx2D(int w, int h) {
	return new AccelVectorCplx2d( w, h);
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
}
