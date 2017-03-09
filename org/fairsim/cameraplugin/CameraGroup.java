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
 *
 * @author m.lachetta
 */
public class CameraGroup {
    private String[] group;
    
    CameraGroup(String name, String[] configs) {
        group = new String[configs.length + 1];
        group[0] = name;
        for (int i = 1; i < group.length; i++) {
            group[i] = configs[i-1];
        }
    }
    
    public CameraGroup(String encodedCameraGroup) {
        group = encodedCameraGroup.split(",");
    }
    
    public String getNmae() {
        return group[0];
    }
    
    public String getConfig(int configId) {
        return group[configId+1];
    }
    
    public String[] getConfigArray() {
        int len = group.length - 1;
        String[] s = new String[len];
        for (int i = 0; i < len; i++) {
            s[i] = group[i+1];
        }
        return s;
    }
    
    String encode() {
        String output = group[0];
        for (int i = 1; i < group.length; i++) {
            output += "," + group[i];
        }
        return output;
    }
}
