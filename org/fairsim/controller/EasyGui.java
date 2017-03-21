/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fairsim.controller;

import java.util.ArrayList;
import java.util.List;
import org.fairsim.utils.Tool;

/**
 *
 * @author Mario
 */
public class EasyGui {
    
    private boolean active;
    private final AdvGui advancedGui;
    private final Ctrl controllerPanel;
    private final Sync syncPanel;
    private final Reg regPanel;
    private final  List<Movie> camGuis;
    private static final List<RunningOrder> runningOrders = new ArrayList<>();
    
    EasyGui(AdvGui aGui) {
        active = false;
        advancedGui = aGui;
        controllerPanel = aGui.getCtrl();
        syncPanel = aGui.getSync();
        regPanel = aGui.getReg();
        camGuis = aGui.getCams();
    }
    
    void activate() {
        ControllerClient.ArduinoRunningOrder[] arduinoRos = null;
        String[] deviceBox = null;
        try {
            arduinoRos = controllerPanel.getArduinoRos();
            deviceBox = controllerPanel.getDeviceRos();
        } catch (NullPointerException ex) {
            Tool.error("Easy to use controller could not be activated", false);
            return;
        }
        
        for(int i = 0; i < arduinoRos.length; i++) {
            String device = arduinoRos[i].name.split("_", 2)[0];
            String name = arduinoRos[i].name.split("_", 2)[1];
            int deviceRo = getDeviceRo(name, deviceBox);
            if (deviceRo < 0) continue;
            boolean allowBigRoi = name.split("_")[1].endsWith("ms");
            runningOrders.add(new RunningOrder(device, name, deviceRo, i,
                    arduinoRos[i].syncDelay, 8000, arduinoRos[i].syncFreq,
                    arduinoRos[i].exposureTime, allowBigRoi));
        }
        active = true;
    }
    
    private int getDeviceRo(String name, String[] deviceRos) {
        for(int i = 0; i < deviceRos.length; i++) {
            if (name.equals(deviceRos[i])) return i;
        }
        return -1;
    }
    
    void startRunningOrder(RunningOrder ro) {
        
    }
    
    void stopRunningOrder() {
        
    }
    
    class RunningOrder {
        final String device, name;
        final int deviceRo, arduinoRo;
        final int syncDelay, syncAvr, syncFreq;
        final int exposureTime;
        final String camGroup = "fastSIM", camString = "fastSIM";
        final boolean allowBigRoi;

        RunningOrder(String device, String name, int deviceRo, int arduinoRo, int sDelay, int sAvr, int sFreq, int eTime, boolean allowBigRoi) {
            this.device = device;
            this.name = name;
            this.deviceRo = deviceRo;
            this.arduinoRo = arduinoRo;
            this.syncDelay = sDelay;
            this.syncAvr = sAvr;
            this.syncFreq = sFreq;
            this.exposureTime = eTime;
            this.allowBigRoi = allowBigRoi;
        }
    }
    
    static interface AdvGui {
        void setRoi(int size);
        Ctrl getCtrl();
        Sync getSync();
        Reg getReg();
        List<Movie> getCams();
    }
    
    static interface Ctrl extends Movie{
        ControllerClient.ArduinoRunningOrder[] getArduinoRos();
        String[] getDeviceRos();
        void takePhoto(RunningOrder ro);
    }
    
    static interface Sync {
        void setSync(int delay, int avr, int freq);
    }
    
    static interface Reg {
        void register(boolean b);
    }
    
    static interface Movie {
        void startMovie(RunningOrder ro);
        void stopMovie();
    }
}
