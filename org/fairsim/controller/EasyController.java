/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fairsim.controller;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Mario
 */
public class EasyController {
    
    private final ControllerPanel controllerGui;
    private final SyncPanel syncGui;
    private final RegistrationPanel regGui;
    private final CameraPanel[] camGuis;
    private static final List<RunningOrder> runningOrders = new ArrayList<>();
    
    EasyController(ControllerPanel controllerGui, SyncPanel syncGui,
            RegistrationPanel regGui, CameraPanel[] camGuis,
            ControllerClient.ArduinoRunningOrder arduinoRos, String[] slmRos) {
        this.controllerGui = controllerGui;
        this.syncGui = syncGui;
        this.regGui = regGui;
        this.camGuis = camGuis;
        
        //TODO
    }
    
    private class RunningOrder {
        final String name;
        final int slmRo, arduinoRo;
        final int syncDelay, syncAvr, syncFreq;
        final int exposureTime;
        final String camGroup = "fastSIM", camString = "fastSIM";
        final boolean allowBigRoi;

        RunningOrder(String name, int slmRo, int arduinoRo, int sDelay, int sAvr, int sFreq, int eTime, boolean allowBigRoi) {
            this.name = name;
            this.slmRo = slmRo;
            this.arduinoRo = arduinoRo;
            this.syncDelay = sDelay;
            this.syncAvr = sAvr;
            this.syncFreq = sFreq;
            this.exposureTime = eTime;
            this.allowBigRoi = allowBigRoi;
        }
    }
}
