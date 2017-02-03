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

import bunwarpj.Transformation;
import bunwarpj.bUnwarpJ_;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import java.io.File;
import java.io.IOException;
import java.util.zip.DataFormatException;
import org.fairsim.fiji.Converter;
import org.fairsim.linalg.Vec2d;
import org.fairsim.livemode.ReconstructionRunner;

/**
 *
 * @author m.lachetta
 */
public class RegFileCreator {
    
    String regFolder;
    ImagePlus targetImp;
    ImagePlus sourceImp;
    ImageProcessor targetMskIP;
    ImageProcessor sourceMskIP;
    int mode;
    int img_subsamp_fact;
    int min_scale_deformation;
    int max_scale_deformation;
    double divWeight;
    double curlWeight;
    double landmarkWeight;
    double imageWeight;
    double consistencyWeight;
    double stopThreshold;
    
    RegFileCreator(String regFolder) {
        this.regFolder = regFolder;
        mode = 2;
        img_subsamp_fact = 0;
        min_scale_deformation = 0;
        max_scale_deformation = 2;
        divWeight = 0;
        curlWeight = 0;
        landmarkWeight = 0;
        imageWeight = 1;
        consistencyWeight = 10;
        stopThreshold = 0.01;
    }
    
    void createRegFile(Vec2d.Real sourceVec, Vec2d.Real targetVec, String targetChannelName) {
        ImagePlus targetImg = Converter.converteVecImg(targetVec, "targetVec");
        ImageProcessor targetProcessor = targetImg.getProcessor();
        ImagePlus sourceImg = Converter.converteVecImg(sourceVec, "sourceVec");
        ImageProcessor sourceProcessor = sourceImg.getProcessor();
        
        System.out.println("[RegFileCreator]: Starting transformation");
        
        Transformation elasticTransf = bUnwarpJ_.computeTransformationBatch(targetImg, sourceImg, targetProcessor,
                sourceProcessor, mode, img_subsamp_fact, min_scale_deformation,
                max_scale_deformation, divWeight, curlWeight, landmarkWeight,
                imageWeight, consistencyWeight, stopThreshold);
        
        System.out.println("[RegFileCreator]: Transformation finished, try to save elastic transformation");
        
        elasticTransf.saveDirectTransformation(regFolder + targetChannelName + "Elastic.txt");
        
        System.out.println("[RegFileCreator]: Elastic Transformation saved, try to save Images");
        
        Converter.saveImage(targetImg, regFolder + "targetImg.tif");
        Converter.saveImage(sourceImg, regFolder + "sourceImg.tif");
        
        System.out.println("[RegFileCreator]: Images saved, try to converte elastic transformation to raw");
        
        new File(regFolder + targetChannelName + ".txt").delete();
        bUnwarpJ_.convertToRawTransformationMacro(regFolder + "targetImg.tif",
                regFolder + "sourceImg.tif", regFolder + targetChannelName + "Elastic.txt",
                regFolder + targetChannelName + ".txt");
        
        System.out.println("[RegFileCreator]: Saved raw transformation, try to delete uselessfiles");
        
        new File(regFolder + targetChannelName + "Elastic.txt").delete();
        new File(regFolder + "targetImg.tif").delete();
        new File(regFolder + "sourceImg.tif").delete();
        
        System.out.println("[RegFileCreator]: Finished all");
    }
    
    void setOptions(int mode, int img_subsamp_fact, int min_scale_deformation,
            int max_scale_deformation, double divWeight, double curlWeight,
            double landmarkWeight, double imageWeight, double consistencyWeight,
            double stopThreshold) throws DataFormatException {
        if (mode >= 0 && mode <= 2 && img_subsamp_fact >= 0 && img_subsamp_fact <= 7 &&
                min_scale_deformation >= 0 && min_scale_deformation <= 3 &&
                max_scale_deformation >= 0 && max_scale_deformation <= 4 &&
                divWeight >= 0 && curlWeight >= 0 && landmarkWeight >= 0 &&
                imageWeight >= 0 && consistencyWeight >= 0 && stopThreshold >= 0) {
            this.mode = mode;
            this.img_subsamp_fact = img_subsamp_fact;
            this.min_scale_deformation = min_scale_deformation;
            this.max_scale_deformation = max_scale_deformation;
            this.divWeight = divWeight;
            this.curlWeight = curlWeight;
            this.landmarkWeight = landmarkWeight;
            this.imageWeight = imageWeight;
            this.consistencyWeight = consistencyWeight;
            this.stopThreshold = stopThreshold;
        } else {
            throw new DataFormatException("One or more values are not compatible");
        }
    }
    
    void createChannelRegFile(int targetId, int sourceId, String targetChannelName ,ReconstructionRunner recRunner) throws DataFormatException {
        if (targetId == sourceId) {
            throw new DataFormatException("Target and source can not be equal");
        }
        
        recRunner.latestReconLock[targetId].lock();
        recRunner.latestReconLock[sourceId].lock();
        try {
            Vec2d.Real targetVec = recRunner.getLatestReconVec(targetId);
            Vec2d.Real sourceVec = recRunner.getLatestReconVec(sourceId);
            createRegFile(targetVec, sourceVec, targetChannelName);
        } finally {
            recRunner.latestReconLock[targetId].unlock();
            recRunner.latestReconLock[sourceId].unlock();
        }
    }
    
    /**
     * for testing
     * @param arg
     * @throws IOException 
     */
    public static void main( String [] arg ) throws IOException {
        //testing creation of registrationFile
        Vec2d.Real targetVec = Converter.loadVec("D:/vigor-registration/testReg.tif");
        Vec2d.Real sourceVec = Converter.loadVec("D:/vigor-registration/testRegShifted.tif");
        RegFileCreator creator = new RegFileCreator("D:/vigor-registration/");
        creator.createRegFile(sourceVec, targetVec, "1234");
        
        //testing registration of registrationFile
        Registration reg = new Registration("D:/vigor-registration/1234.txt");
        Vec2d.Real registeredSourceVec = reg.registerReconImage(sourceVec);
        Converter.saveVec(registeredSourceVec, "D:/vigor-registration/registeredTestRegShifted.tif");
    }
}
