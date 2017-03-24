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
package org.fairsim.cameraplugin;

/**
 * Helpful class to handle camera groups and configs
 * @author m.lachetta
 */
public class CameraGroup {
    private String[] group;
    
    /**
     * Constructor
     * @param name name of this group
     * @param configs array of configs in this group
     */
    CameraGroup(String name, String[] configs) {
        group = new String[configs.length + 1];
        group[0] = name;
        for (int i = 1; i < group.length; i++) {
            group[i] = configs[i-1];
        }
    }
    
    /**
     * decodes a CameraGroup and creates a new instance
     * @param encodedCameraGroup the string encoded camera group
     */
    public CameraGroup(String encodedCameraGroup) {
        group = encodedCameraGroup.split(",");
    }
    
    /**
     * 
     * @return name of this group
     */
    public String getNmae() {
        return group[0];
    }
    
    /**
     * 
     * @param configId id of the specified config
     * @return the specified config
     */
    public String getConfig(int configId) {
        return group[configId+1];
    }
    
    /**
     * 
     * @return an array of all configs in this group
     */
    public String[] getConfigArray() {
        int len = group.length - 1;
        String[] s = new String[len];
        for (int i = 0; i < len; i++) {
            s[i] = group[i+1];
        }
        return s;
    }
    
    /**
     * 
     * @return an encoded string of this group
     */
    String encode() {
        String output = group[0];
        for (int i = 1; i < group.length; i++) {
            output += "," + group[i];
        }
        return output;
    }
}
