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
package org.fairsim.controller;

/**
 *
 * @author m.lachetta
 */
public class Utilities {
    private Utilities() {}
    
    public static String encodeArray(String prefix, String[] array) {
        int len = array.length;
        String output = prefix;
        if (len == 0) {
            return output;
        } else {
            for (int i = 0; i < len; i++) {
                output += ";" + array[i];
            }

            return output;
        }
    }
    
    public static String[] decodeArray(String encodedArray) {
        String[] split = encodedArray.split(";");
        String[] data = new String[split.length - 1];
        for (int i = 0; i < data.length; i++) {
            data[i] = split[i + 1];
        }
        return data;
    }
}
