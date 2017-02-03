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
package org.fairsim.fiji;

import ij.ImagePlus;
import ij.io.FileSaver;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.fairsim.linalg.BasicVectors;
import org.fairsim.linalg.Vec2d;

/**
 *
 * @author m.lachetta
 */
public class Converter {
    /*
    int width, height;
    
    private Converter(Vec2d.Real vec) {
        width = vec.vectorWidth();
        height = vec.vectorHeight();
    }
    
    private Converter(ImagePlus image) {
        width = image.getWidth();
        height = image.getHeight();
    }
    
    public Converter(int width, int height) {
        this.width = width;
        this.height = height;
    }
    */
    public static ImagePlus converteVecImg(Vec2d.Real vec, String imageName) {
        int width = vec.vectorWidth();
        int height = vec.vectorHeight();
        ImageProcessor processor = new FloatProcessor(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                processor.setf(x, y, vec.get(x, y)); 
            }
        }
        
        return new ImagePlus(imageName, processor);
    }
    
    public static ImagePlus converteVecImg(Vec2d.Real vec) {
        return converteVecImg(vec, "convertedVector");
    }
    
    public static void saveImage(ImagePlus image, String file) {
        new FileSaver(image).saveAsTiff(file);
    }
    
    public static ImagePlus loadImage(String file) {
        return new ImagePlus(file);
    }    
    
    public static Vec2d.Real comverteImgVec (ImagePlus image) {
        int width = image.getWidth();
        int height = image.getHeight();
        ImageProcessor processor = image.getProcessor();
        Vec2d.Real vec = BasicVectors.getFactory().createReal2D(width, height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                vec.set(x, y, processor.get(x, y));
            }
        }
        return vec;
    }
    
    public static void saveVec(Vec2d.Real vec, String file) {
        ImagePlus image = converteVecImg(vec);
        saveImage(image, file);
    }
    
    public static Vec2d.Real loadVec(String file) {
        ImagePlus image = loadImage(file);
        return comverteImgVec(image);
    }
    
    /*
    public static void main( String [] arg ) {
        saveVec( loadVec("D:/vigor-registration/testReg.tif"), "D:/vigor-registration/wrapped.tif" );
    }
    */
}
