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

package org.fairsim.registration;

import ij.ImagePlus;
import ij.io.FileSaver;
import ij.process.ImageProcessor;
import java.io.IOException;
import java.util.zip.DataFormatException;
import org.fairsim.fiji.ImageVector;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.linalg.BasicVectors;
import org.fairsim.linalg.Vec2d;

/**
 * Class for testing and timing the Registrationprocess
 * @author Mario
 */
public class TestReg {
    static long imageNr = 0;
    
    public static void diskWriter(Vec2d.Real img)  {
        ImageVector iv = ImageVector.create(img.vectorWidth(), img.vectorHeight());
        iv.copy(img);
        ImageProcessor imgProcessor = iv.img();
        
        ImagePlus imgPlus = new ImagePlus("to save", imgProcessor);
        new FileSaver(imgPlus).saveAsTiff("D:/vigor-tmp/save/" + imageNr + ".tif");
        
        
        imageNr++;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, DataFormatException {
        ImagePlus imgPlus = new ImagePlus("imageRegs/Widefield_original.jpg");
        imgPlus.setSlice(-1);
        ImageProcessor imgProcessor = imgPlus.getProcessor();

        VectorFactory vf = BasicVectors.getFactory();
        Vec2d.Real source = vf.createReal2D(imgProcessor.getWidth(), imgProcessor.getHeight());

        for (int y = 0; y < imgProcessor.getHeight(); y++) {
            for (int x = 0; x < imgProcessor.getWidth(); x++) {
                source.set(x, y, imgProcessor.get(x, y));
            }
        }
        Registration reg = new Registration("imageRegs/widefield.txt");
        Registration.setRecon(true);
        if (Registration.isRecon()) {
            
            Vec2d.Real regVec = reg.registerImageOld(source, 'r');

            for (int y = 0; y < imgProcessor.getHeight(); y++) {
                for (int x = 0; x < imgProcessor.getWidth(); x++) {
                    imgProcessor.set(x, y, (int) regVec.get(x, y));
                }
            }
        }

        new FileSaver(imgPlus).saveAsJpeg("imageRegs/Widefield_registered.jpg");


        
        //timing for 1000 registrations
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < 1000; i++) {
            reg.registerImageOld(source, 'r');
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Time for 1000  'OLD' registrations needed: " + (endTime - startTime) + " ms");
        
        startTime = System.currentTimeMillis();
        for (int i = 0; i < 1000; i++) {
            reg.registerReconImageNew(source);
        }
        endTime = System.currentTimeMillis();
        System.out.println("Time for 1000  'NEW' registrations needed: " + (endTime - startTime) + " ms");
        
        startTime = System.currentTimeMillis();
        for (int i = 0; i < 1000; i++) {
            reg.registerReconImageInverse(source);
        }
        endTime = System.currentTimeMillis();
        System.out.println("Time for 1000  'Inverse' registrations needed: " + (endTime - startTime) + " ms");
         
    }

}
