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
package org.fairsim.transport;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

/**
 *
 * @author m.lachetta
 */
public class LiveStackConverter {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("# Usage:\n\tFolder\n\tOmero-identifier");
            System.exit(1);
        }
        File dir = new File(args[0]);
        File[] foundFiles;
        foundFiles = dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.startsWith(args[1]) && (name.endsWith(".livestack") || name.endsWith(".livesim"));
            }
        });
        System.out.println("found " + foundFiles.length + " files");
        if (foundFiles.length < 1) {
            System.out.println("No files found?");
            System.exit(2);
        }
        System.out.println(foundFiles[0]);
        System.out.println("opening " + foundFiles[0].getAbsolutePath());
        System.out.println("done");
        LiveStack ls = new LiveStack(foundFiles[0].getAbsolutePath());
        String outFile = foundFiles[0].getAbsolutePath() + ".tif";
        ls.saveAsTiff(outFile);
        System.out.println("saved as: " + outFile);
    }
    
}
