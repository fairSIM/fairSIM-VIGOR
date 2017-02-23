/*
 * Code to communicate with the arduino.
 * This is a modifyed class of the "Sample Java Code" from "playground.arduino.cc"
 * http://playground.arduino.cc/Interfacing/Java
 * Original code under CC-BY-SA-3.0
 */
package org.fairsim.controller;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import gnu.io.CommPortIdentifier;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Enumeration;

/**
 * Class to controll the Arduino.
 * 
 * @author m.lachetta
 */
public class ArduinoController implements SerialPortEventListener {

    SerialPort serialPort;
    ControllerServerGui serverGui;
    /**
     * The port we're normally going to use.
     */
    private static final String PORT_NAMES[] = {
        "/dev/tty.usbserial-A9007UX1", // Mac OS X
        "/dev/ttyACM0", // Raspberry Pi
        "/dev/ttyUSB0", // Linux
        "COM3", // Windows
    };
    /**
     * A BufferedReader which will be fed by a InputStreamReader converting the
     * bytes into characters making the displayed results codepage independent
     */
    private BufferedReader input;
    /**
     * The output stream to the port
     */
    private OutputStream output;
    /**
     * Milliseconds to block while waiting for port open
     */
    private static final int TIME_OUT = 2000;
    /**
     * Default bits per second for COM port.
     */
    private static final int DATA_RATE = 19200;

    private ArduinoController() throws Exception{
        initialize();
    }
    
    ArduinoController(ControllerServerGui serverGui) {
        this.serverGui = serverGui;
    }
    

    private void initialize() throws Exception {
        CommPortIdentifier portId = null;
        Enumeration portEnum = CommPortIdentifier.getPortIdentifiers();

        //First, Find an instance of serial port as set in PORT_NAMES.
        while (portEnum.hasMoreElements()) {
            CommPortIdentifier currPortId = (CommPortIdentifier) portEnum.nextElement();
            for (String portName : PORT_NAMES) {
                if (currPortId.getName().equals(portName)) {
                    portId = currPortId;
                    break;
                }
            }
        }
        if (portId == null) {
            String massage = "Could not find COM port.";
            System.out.println(massage);
            throw new UnsupportedOperationException(massage);
        }

            // open serial port, and use class name for the appName.
            serialPort = (SerialPort) portId.open(this.getClass().getName(), TIME_OUT);

            // set port parameters
            serialPort.setSerialPortParams(DATA_RATE, SerialPort.DATABITS_8, SerialPort.STOPBITS_1, SerialPort.PARITY_NONE);

            // open the streams
            input = new BufferedReader(new InputStreamReader(serialPort.getInputStream()));
            output = serialPort.getOutputStream();

            // add event listeners
            serialPort.addEventListener(this);
            serialPort.notifyOnDataAvailable(true);
    }

    /**
     * This should be called when you stop using the port. This will prevent
     * port locking on platforms like Linux.
     */
    public synchronized void close() {
        if (serialPort != null) {
            serialPort.removeEventListener();
            serialPort.close();
        }
    }

    /**
     * Handle an event on the serial port. Read the data and print it.
     */
    public synchronized void serialEvent(SerialPortEvent oEvent) {
        if (oEvent.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
            try {
                String inputLine = input.readLine();
                try {
                    serverGui.showText("Arduino: " + inputLine);
                } catch (NullPointerException ex) {
                    System.out.println("Arduino: " + inputLine);
                }
            } catch (Exception e) {
                System.err.println("Arduino: " + e.toString());
                //close();
                try {
                    serverGui.showText("Arduino: Error: " + e.toString());
                } catch (NullPointerException ex) {}
            }
        }
        // Ignore all the other eventTypes, but you should consider the other ones.
    }

    synchronized String sendCommand(String input) {
        byte[] command = input.getBytes(Charset.forName("UTF-8"));
        try {
            output.write(command);
            output.flush();
            return "Command '" + new String(command) + "' transmitted to the arduino";
        } catch (NullPointerException ex) {
            return "Error: No Connection to the arduino";
        } catch (Exception ex) {
            return "Error: " + ex.toString();
        }
    }
    
    synchronized String connect() {
        try {
            initialize();
            return "Connected to the arduino";
        } catch (Exception e) {
            return "Error: " + e.toString();
        }
    }
    
    synchronized String disconnect() {
        try {
            close();
            return "Disconnected from the arduino";
        } catch (Exception e) {
            return "Error: " + e.toString();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        try{
            ArduinoController main = new ArduinoController(null);
            main.connect();
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            String input;
            while (true) {
                System.out.println("Command for the Microcontroller: ");
                input = in.readLine();
                main.sendCommand(input);
            }
            //main.close();
        } catch (Exception ex) {
            System.err.println(ex);
        }
    }
}
