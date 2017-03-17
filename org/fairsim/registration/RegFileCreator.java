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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.zip.DataFormatException;
import org.fairsim.fiji.Converter;
import org.fairsim.linalg.Vec2d;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.utils.Tool;

/**
 * Class for building new registration files, supported by BUnwarpJ
 * @author m.lachetta
 */
public class RegFileCreator {
    
    final String regFolder;
    final String[] channelNames;
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
    private static final int REGISTRATIONSIZE = 1024;
    
    /**
     * Constructor
     * @param regFolder Folder for reading/saving files
     * @param channelNames String array of live mode channels
     */
    RegFileCreator(String regFolder, String[] channelNames) {
        this.regFolder = regFolder;
        this.channelNames = channelNames;
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
    
    /**
     * Creates a new raw registration file with the BUnwarpJ algorithm and saves it
     * @param sourceVec source image for registration with BUnwarpJ
     * @param targetVec target image for registration with BUnwarpJ
     * @param targetChannelName  channel which the registration is for
     */
    void createRegFile(Vec2d.Real sourceVec, Vec2d.Real targetVec, String targetChannelName) {
        ImagePlus targetImg = Converter.converteVecImg(targetVec, "targetVec");
        ImageProcessor targetProcessor = targetImg.getProcessor();
        ImagePlus sourceImg = Converter.converteVecImg(sourceVec, "sourceVec");
        ImageProcessor sourceProcessor = sourceImg.getProcessor();
        
        
        Transformation elasticTransf = bUnwarpJ_.computeTransformationBatch(targetImg, sourceImg, targetProcessor,
                sourceProcessor, mode, img_subsamp_fact, min_scale_deformation,
                max_scale_deformation, divWeight, curlWeight, landmarkWeight,
                imageWeight, consistencyWeight, stopThreshold);
        
        
        elasticTransf.saveDirectTransformation(Tool.getFile(regFolder + targetChannelName + "Elastic.txt").getAbsolutePath());
        
        
        Converter.saveImage(targetImg, Tool.getFile(regFolder + "targetImg.tif").getAbsolutePath());
        Converter.saveImage(sourceImg, Tool.getFile(regFolder + "sourceImg.tif").getAbsolutePath());
        
        
        Tool.getFile(regFolder + targetChannelName + ".txt").delete();
        bUnwarpJ_.convertToRawTransformationMacro(Tool.getFile(regFolder + "targetImg.tif").getAbsolutePath(),
                Tool.getFile(regFolder + "sourceImg.tif").getAbsolutePath(), Tool.getFile(regFolder + targetChannelName + "Elastic.txt").getAbsolutePath(),
                Tool.getFile(regFolder + targetChannelName + ".txt").getAbsolutePath());
        
        
        Tool.getFile(regFolder + targetChannelName + "Elastic.txt").delete();
        Tool.getFile(regFolder + "targetImg.tif").delete();
        Tool.getFile(regFolder + "sourceImg.tif").delete();
    }
    
    /**
     * Sets fields for registering of this class.
     * Names are chosen equal to the BUnwarpJ GUI.
     * @throws DataFormatException if input values are impossible to set
     */
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
    
    /**
     * Creates a registration file in live mode
     * @param targetId id of target image
     * @param sourceId is of source image
     * @param recRunner reconstruction ReconstructionRunner of live mode
     * @throws DataFormatException if target- and source-id are equal
     * @throws IOException if copying goes wrong
     */
    void createChannelRegFile(int targetId, int sourceId, ReconstructionRunner recRunner) throws DataFormatException, IOException {
        if (targetId == sourceId) {
            throw new DataFormatException("Target and source can not be equal");
        }
        
        String targetChannelName = channelNames[targetId];
        String sourcChannelName = channelNames[sourceId];
        
        Vec2d.Real targetVec;
        Vec2d.Real sourceVec;
        
        recRunner.latestReconLock[targetId].lock();
        recRunner.latestReconLock[sourceId].lock();
        try {
            targetVec = recRunner.getLatestReconVec(targetId);
            sourceVec = recRunner.getLatestReconVec(sourceId);
        } finally {
            recRunner.latestReconLock[targetId].unlock();
            recRunner.latestReconLock[sourceId].unlock();
        }
        
        if (targetVec.vectorWidth() != REGISTRATIONSIZE || targetVec.vectorHeight() != REGISTRATIONSIZE
                || sourceVec.vectorWidth() != REGISTRATIONSIZE
                || sourceVec.vectorHeight() != REGISTRATIONSIZE) {
            throw new DataFormatException("Need pixel size: " + REGISTRATIONSIZE);
        }
        
        createRegFile(targetVec, sourceVec, targetChannelName);

        Path targetPath = Paths.get(Tool.getFile(regFolder + targetChannelName + ".txt").getAbsolutePath());
        Path sourcePath = Paths.get(Tool.getFile(regFolder + targetChannelName + "to" + sourcChannelName + "-" + Tool.readableTimeStampMillis(System.currentTimeMillis(), false) + ".txt").getAbsolutePath());
        Files.copy(targetPath, sourcePath, StandardCopyOption.REPLACE_EXISTING);
    }
    
    /**
     * deletes the registration file for a specified channel
     * @param channelName channel of the registration file that should be deleted
     * @return only true if the to deleted file exists and was successfully deleted
     */
    boolean deleteRegFile(String channelName) {
        return Tool.getFile(regFolder + channelName + ".txt").delete();
    }
}
