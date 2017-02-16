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

package org.fairsim.deconvolution;

import org.fairsim.fiji.DisplayWrapper;
import org.fairsim.linalg.BasicVectors;
import org.fairsim.linalg.Transforms;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.utils.ImageDisplay;

public class Utilities {
    
    static Vec2d.Cplx convolve(Vec2d.Cplx vec1, Vec2d.Cplx vec2) {
        // real space -> fourier space
        vec1.fft2d(false);
        Transforms.swapQuadrant(vec2);
        vec2.fft2d(false);
        // elementwise multiplication
        vec1.times(vec2);
        // fourier space -> real space
        vec1.fft2d(true);
        
        return vec1;
    }
    
    static void displayVector(Vec2d.Cplx vec) {
        VectorFactory vf = BasicVectors.getFactory();
        // make vec real
        Vec2d.Real vecReal = vf.createReal2D(vec.vectorWidth(), vec.vectorHeight());
        vecReal.copy(vec.duplicateReal());
        
        displayVector(vecReal);
    }
    
    static void displayVector(Vec2d.Real vec) {
        ImageDisplay imdisp  = new DisplayWrapper(vec.vectorWidth(), vec.vectorHeight(), "Untitled");
        imdisp.addImage(vec, "untitled");
        imdisp.display();
    }
}
