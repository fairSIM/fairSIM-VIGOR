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
import static org.fairsim.transport.LiveStack.open;

/**
 *
 * @author m.lachetta
 */
public class LiveSimTif2Meta {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length != 2) {
            System.out.println("# Usage:\n\tFolder\n\tname");
            System.exit(1);
        }
        File dir = new File(args[0]);
        File[] foundFiles;
        foundFiles = dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.startsWith(args[1]) && (name.endsWith(".livestack.tif") || name.endsWith(".livesim.tif"));
            }
        });
        System.out.println("found " + foundFiles.length + " files");
        if (foundFiles.length < 1) {
            System.out.println("No files found?");
            System.exit(2);
        }
        if (foundFiles.length < 1) {
            System.out.println("Filename ambiguous");
            System.exit(2);
        }
        String filename = foundFiles[0].getAbsolutePath();
        System.out.println("opening " + filename);
        System.out.println("done");
        LiveStack ls;
        ls = open(filename);
        ls.liveStackToMeta(filename);
        System.out.println("done");
    }
    
}
