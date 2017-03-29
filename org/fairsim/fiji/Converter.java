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
import ij.ImageStack;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import org.fairsim.linalg.BasicVectors;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.Vec3d;
import org.fairsim.linalg.VectorFactory;

/**
 * Converter between fairSIM vectors and image files.
 */
public class Converter {
    // Marcels Kram
    
    /**
     * Create complex 2D vector from image file.
     * 
     * @param fileLocation image location
     * @param vf vector factory to create the vector
     */
    public static Vec2d.Cplx fileToVec2dCplx(String fileLocation, VectorFactory vf) {
        Opener opener = new Opener();
	ImagePlus imp = opener.openImage(fileLocation);
        
        final int w = imp.getWidth();
        final int h = imp.getHeight();
        
        Vec2d.Real vecReal = vf.createReal2D(w, h);
        ImageProcessor ip = imp.getStack().getProcessor(1).convertToFloat();
        float[] values = (float[]) ip.getPixels();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                vecReal.set(x, y, values[y*w + x]);
            }
        }

        Vec2d.Cplx vecCplx = vf.createCplx2D(w, h);
        vecCplx.copy(vecReal);
        return vecCplx;
    }

    /**
     * Create complex 3D vector from image file.
     * 
     * @param fileLocation image location
     * @param vf vector factory to create the vector
     */
    public static Vec3d.Cplx fileToVec3dCplx(String fileLocation, VectorFactory vf) {
        Opener opener = new Opener();
	ImagePlus imp = opener.openImage(fileLocation);
        
        final int w = imp.getWidth();
        final int h = imp.getHeight();
        final int d = imp.getStackSize();
        
        Vec3d.Real vecReal = vf.createReal3D(w, h, d);
        for (int z = 0; z < d; z++) {
            ImageProcessor ip = imp.getStack().getProcessor(z + 1).convertToFloat();
            float[] values = (float[]) ip.getPixels();
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    vecReal.set(x, y, z, values[y*w + x]);
                }
            }
        }
        Vec3d.Cplx vecCplx = vf.createCplx3D(w, h, d);
        vecCplx.copy(vecReal);
        return vecCplx;
    }

    /**
     * Create and save TIFF file from complex 2D vector.
     * Only the real part of the vector is used.
     * 
     * @param vec input vector
     * @param fileLocation location for saving file
     */
    public static void vec2dCplxToFile(Vec2d.Cplx vec, String fileLocation) {
        final int w = vec.vectorWidth();
        final int h = vec.vectorHeight();

        float[] values = new float[w*h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                values[y*w + x] = vec.get(x, y).re;
            }
        }
        ImageProcessor ip = new FloatProcessor(w, h, values);

        ImagePlus img = new ImagePlus("", ip);
        new FileSaver(img).saveAsTiff(fileLocation);
    }

    /**
     * Create and save TIFF file from complex 3D vector.
     * Only the real part of the vector is used.
     * 
     * @param vec input vector
     * @param fileLocation location for saving file
     */
    public static void vec3dCplxToFile(Vec3d.Cplx vec, String fileLocation) {
        final int w = vec.vectorWidth();
        final int h = vec.vectorHeight();
        final int d = vec.vectorDepth();
        ImageStack stack = new ImageStack(w, h);
        
        for(int z = 0; z < d; z++) {
            float[] values = new float[w*h];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    values[y*w + x] = vec.get(x, y, z).re;
                }
            }
            ImageProcessor ip = new FloatProcessor(w, h, values);
            stack.addSlice(ip);
        }

        ImagePlus imp = new ImagePlus("", stack);
        
        // convert imageplus object from 32bit to 16bit
        ImageConverter c = new ImageConverter(imp);
        c.convertToGray16();
        
        new FileSaver(imp).saveAsTiff(fileLocation);
    }
    
    
    
    
    
    
    

    // Marios Kram
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

    public static Vec2d.Real comverteImgVec(ImagePlus image) {
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
