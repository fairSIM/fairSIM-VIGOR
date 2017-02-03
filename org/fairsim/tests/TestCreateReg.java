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
package org.fairsim.tests;

import bunwarpj.*;
import ij.ImagePlus;
import ij.io.FileSaver;
import ij.process.ImageProcessor;
import java.io.IOException;
import org.fairsim.linalg.BasicVectors;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.registration.Registration;

/**
 *
 * @author m.lachetta
 */
public class TestCreateReg {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        ImagePlus sourceImg = new ImagePlus("D:/vigor-registration/Recon_shifted.jpg");
        ImageProcessor sourceProcessor = sourceImg.getProcessor();
        ImagePlus targetImg = new ImagePlus("D:/vigor-registration/Recon_original.jpg");
        ImageProcessor targetProcessor = targetImg.getProcessor();
                
        Transformation testTransformation;
        testTransformation = bUnwarpJ_.computeTransformationBatch(targetImg, sourceImg, targetProcessor, sourceProcessor, 0, 0, 0, 0, 0, 0, 0, 1, 10, 0.01);
        testTransformation.saveDirectTransformation("D:/vigor-registration/regTest.txt");
        bUnwarpJ_.convertToRawTransformationMacro("D:/vigor-registration/Recon_original.jpg", "D:/vigor-registration/Recon_shifted.jpg", "D:/vigor-registration/regTest.txt", "D:/vigor-registration/regTestRaw.txt");
        
        Registration reg = new Registration("D:/vigor-registration/regTestRaw.txt");
        
        VectorFactory vf = BasicVectors.getFactory();
        Vec2d.Real sourceVec = vf.createReal2D(sourceProcessor.getWidth(), sourceProcessor.getHeight());

        for (int y = 0; y < sourceProcessor.getHeight(); y++) {
            for (int x = 0; x < sourceProcessor.getWidth(); x++) {
                sourceVec.set(x, y, sourceProcessor.get(x, y));
            }
        }
        
            
        Vec2d.Real regVec = reg.registerReconImage(sourceVec);

        for (int y = 0; y < sourceProcessor.getHeight(); y++) {
            for (int x = 0; x < sourceProcessor.getWidth(); x++) {
                sourceProcessor.set(x, y, (int) regVec.get(x, y));
            }
        }

        new FileSaver(sourceImg).saveAsJpeg("D:/vigor-registration/Recon_registered.jpg");
    }
    
}
