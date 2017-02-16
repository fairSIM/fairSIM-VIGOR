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

import ij.ImagePlus;
import ij.process.ImageProcessor;
import org.fairsim.linalg.BasicVectors;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.Vec;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.utils.SimpleMT;

public class DeconvolutionTesting {

    public static void main(String[] args) {
        VectorFactory vf = BasicVectors.getFactory();
        /*
        // image to vec2d
        ImagePlus imgPlus_original = new ImagePlus("C:\\test\\original.tif");
        ImageProcessor imgProcessor_original = imgPlus_original.getProcessor();

        Vec2d.Cplx original = vf.createCplx2D(imgProcessor_original.getWidth(), imgProcessor_original.getHeight());

        for (int y = 0; y < imgProcessor_original.getHeight(); y++) {
            for (int x = 0; x < imgProcessor_original.getWidth(); x++) {
                original.set(x, y, new org.fairsim.linalg.Cplx.Double(imgProcessor_original.get(x, y)));
            }
        }

        // psf to vec2d
        ImagePlus imgPlus_psf = new ImagePlus("C:\\test\\psf2.tif");
        ImageProcessor imgProcessor_psf = imgPlus_psf.getProcessor();

        Vec2d.Cplx psf = vf.createCplx2D(imgProcessor_psf.getWidth(), imgProcessor_psf.getHeight());

        for (int y = 0; y < imgProcessor_psf.getHeight(); y++) {
            for (int x = 0; x < imgProcessor_psf.getWidth(); x++) {
                psf.set(x, y, new org.fairsim.linalg.Cplx.Double(imgProcessor_psf.get(x, y)));
            }
        }
        
        Vec2d.Cplx ergebnis;
        ergebnis = Utilities.convolve(original, psf);
        */
        //Utilities.displayVector(ergebnis);
        
        Vec2d.Real vec1 = vf.createReal2D(2, 2);
        vec1.set(0, 0, 1.0f);
        vec1.set(0, 1, 2.0f);
        vec1.set(1, 0, 3.0f);
        vec1.set(1, 1, 4.0f);
        Vec2d.Real vec2 = vf.createReal2D(2, 2);
        vec2.set(0, 0, 5.0f);
        vec2.set(0, 1, 6.0f);
        vec2.set(1, 0, 7.0f);
        vec2.set(1, 1, 8.0f);
        
        vec1.times(vec2);
        System.out.println(vec1.get(0,0));
        System.out.println(vec1.get(0,1));
        System.out.println(vec1.get(1,0));
        System.out.println(vec1.get(1,1));
        vec1.elementwiseDivision(vec2);
        System.out.println(vec1.get(0,0));
        System.out.println(vec1.get(0,1));
        System.out.println(vec1.get(1,0));
        System.out.println(vec1.get(1,1));
       
        
        // ausgabe als tif
        /*
        Vec2d.Real ausgabe = vf.createReal2D(original.vectorWidth(), original.vectorHeight());
        ausgabe.copy(ergebnis.duplicateReal());
        Converter.saveVec(ausgabe, "C:\\test\\ausgabe.tif");*/
    }
}
