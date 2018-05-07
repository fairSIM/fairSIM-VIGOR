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

package org.fairsim.utils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import gnu.io.CommPortIdentifier;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import gnu.io.UnsupportedCommOperationException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.TooManyListenersException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Class to communicate with a serial device
 *
 * @author m.lachetta
 */
public class SerialCommunicator {

    private SerialPort serialPort;
    private final BlockingQueue<String> serialInput;
    private final BlockingQueue<String> serialOutput;
    private SendingThread sendingThread;
    
    
    /**
     * constructor
     */
    public SerialCommunicator() {
        serialInput = new LinkedBlockingQueue<>();
        serialOutput = new LinkedBlockingQueue<>();
    }
    
    /**
     * 
     * @return list of all accessible comm ports
     */
    public static List<String> getCommPortNames() {
        Enumeration commPorts = CommPortIdentifier.getPortIdentifiers();
        List<String> ports = new ArrayList<>();
         while (commPorts.hasMoreElements()) {
            CommPortIdentifier currPortId = (CommPortIdentifier) commPorts.nextElement();
            ports.add(currPortId.getName());
        }
         return ports;
    }    
    
    /**
     * 
     * @param timeout in milli seconds
     * @return the next message from the serial device or null if timed out
     * @throws InterruptedException if the poll command gets interrupted
     * @throws IOException if a communication error occurs
     */
    public String pollSerialInput(int timeout) throws InterruptedException, IOException {
        String input = serialInput.poll(timeout, TimeUnit.MILLISECONDS);
        return handleMessage(input);
    }
    
    /**
     * 
     * @return the next message from the serial device or null if there is no message
     * @throws IOException if a communication error occurs
     */
    public String pollSerialInput() throws IOException {
        String input = serialInput.poll();
        return handleMessage(input);
    }
    
    /**
     * 
     * @return the next message from the serial device or blocks until there is a massage
     * @throws InterruptedException if the take command gets interrupted
     * @throws IOException IOException if a communication error occurs
     */
    public String takeSerialInput() throws InterruptedException, IOException {
        String input = serialInput.take();
        return handleMessage(input);
    }
    
    /**
     * distinguishes between messages and errors
     * @param input input of the serialInput
     * @return the message if there is a message
     * @throws IOException if there is an error
     */
    private String handleMessage(String input) throws IOException {
        if (input != null) {
            if (input.startsWith("Message: ")) {
                return input.split(": ", 2)[1];
            } else if (input.startsWith("Error: ")) {
                throw new IOException(input.split(": ", 2)[1]);
            } else throw new RuntimeException("This should never happen");
        }
        else return null;
    }
    
    /**
     * Builds up a connection to a serial device.
     * take the first found comm port name. Possible port names:
     * "/dev/tty.usbserial-A9007UX1", // Mac OS X
     * "/dev/ttyACM0", // Raspberry Pi
     * "/dev/ttyUSB0", // Linux
     * "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", // Windows
     * @param baudRate e.g. 9600
     * @throws java.io.IOException
     */
    public void connect(int baudRate) throws IOException {
        String portNames[] = {
            "/dev/tty.usbserial-A9007UX1", // Mac OS X
            "/dev/ttyACM0", // Raspberry Pi
            "/dev/ttyUSB0", // Linux
            "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", // Windows
        };
        CommPortIdentifier portId = null;
        Enumeration portEnum = CommPortIdentifier.getPortIdentifiers();

        //First, Find an instance of serial port as set in PORT_NAMES.
        while (portEnum.hasMoreElements()) {
            CommPortIdentifier currPortId = (CommPortIdentifier) portEnum.nextElement();
            for (String portName : portNames) {
                if (currPortId.getName().equals(portName)) {
                    portId = currPortId;
                    break;
                }
            }
        }
        if (portId == null) throw new IOException("Could not find COM port.");
        
        String portName = portId.getName();
        connect(portName, baudRate);
    }    
    
    /**
     * Builds up a connection to a serial device.
     * @param portName e.g. "COM0"
     * @param baudRate e.g. 9600
     * @throws IOException 
     */
    public void connect(String portName, int baudRate) throws IOException {
        int timeout = 2000;
        connect(portName, baudRate, timeout);
    }
    
    /**
     * Builds up a connection to a serial device.
     * @param portName e.g. "COM0"
     * @param baudRate e.g. 9600
     * @param timeout for building up the connection in ms
     * @throws IOException 
     */
    public void connect(String portName, int baudRate, int timeout) throws IOException {
        int databits = SerialPort.DATABITS_8;
        int stopbits = SerialPort.STOPBITS_1;
        int parity = SerialPort.PARITY_NONE;
        connect(portName, baudRate, timeout, databits, stopbits, parity);
    }
    
    /**
     * Builds up a connection to a serial device.
     * @param portName e.g. "COM0"
     * @param baudRate e.g. 9600
     * @param timeout for building up the connection in ms
     * @param databits gnu.io.SerialPort.DATABITS_?
     * @param stopbits gnu.io.SerialPort.STOPBITS_?
     * @param parity gnu.io.SerialPort.PARITY_...
     * @throws IOException 
     */
    public void connect(String portName, int baudRate, int timeout, int databits, int stopbits, int parity) throws IOException {
        // checks portName
        CommPortIdentifier port = null;
        Enumeration portEnum = CommPortIdentifier.getPortIdentifiers();
        while (portEnum.hasMoreElements()) {
            CommPortIdentifier currPortId = (CommPortIdentifier) portEnum.nextElement();
            if (currPortId.getName().equals(portName)) {
                port = currPortId;
                break;
            }
        }
        if (port == null) throw new IOException("Selected comm port not found: " + portName);
        
        // opens serial port with this class name as owner and sets parameters
        try {
            serialPort = (SerialPort) port.open(this.getClass().getName(), timeout);
            serialPort.setSerialPortParams(baudRate, databits, stopbits, parity);
        } catch (PortInUseException | UnsupportedCommOperationException ex) {
            throw new IOException(ex);
        }
        

        // open the input and output stream
        BufferedReader input = new BufferedReader(new InputStreamReader(serialPort.getInputStream()));
        OutputStream output = serialPort.getOutputStream();

        // add event listener for receiving messages from the serial device
        SerialPortEventListener serialListener = new SerialPortEventListener() {
            @Override
            public synchronized void serialEvent(SerialPortEvent oEvent) {
                if (oEvent.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
                    try {
                        String inputLine = input.readLine();
                        serialInput.add("Message: " + inputLine);
                    } catch (Exception e) {
                        // ecception handling
                        if (e instanceof IOException && e.getMessage().equals("Underlying input stream returned zero bytes")) {
                            //happens sometimes but seams not to be important, will be ignorred
                        } else {
                            serialInput.add("Error: " + e.getMessage());
                        }
                    }
                }
                // Ignore all the other eventTypes, but you should consider the other ones.
            }
        };
        try {
            serialPort.addEventListener(serialListener);
        } catch (TooManyListenersException ex) {
            throw new IOException(ex);
        }
        serialPort.notifyOnDataAvailable(true);
        
        // starts sending thread
        sendingThread = new SendingThread(output);
        sendingThread.start();
    }

    /**
     * closes the connection to the serial device
     */
    public void disconnect() {
        try {
            if (sendingThread != null) sendingThread.interrupt();
            serialOutput.clear();
            serialInput.clear();
            if (sendingThread != null) sendingThread.join();
            
            if (serialPort != null) {
                serialPort.removeEventListener();
                serialPort.close();
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Do not interrupt this");
        }
    }
    
    /**
     * sends a message to the serial device
     * @param message message for the serial device
     */
    public void sendMessage(String message) {
        serialOutput.add(message);
    }

    /**
     * main method for testing direct serial communication with the serial devices
     */
    public static void main(String[] args) throws IOException, InterruptedException {
        System.out.println(getCommPortNames());
        
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String input;
        try {
            SerialCommunicator sc1 = new SerialCommunicator();
            sc1.connect("COM3", 19200);
            SerialCommunicator sc2 = new SerialCommunicator();
            sc2.connect("COM4", 9600);
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        while (true) {
                            System.out.println("Serial 1: " + sc1.takeSerialInput() );
                        }
                    } catch (Exception ex) {
                        System.err.println(ex);
                    }
                }
            }).start();
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        while (true) {
                            System.out.println("Serial 2: " + sc2.takeSerialInput() );
                        }
                    } catch (Exception ex) {
                        System.err.println(ex);
                    }
                }
            }).start();
            while (true) {
                System.out.println("1: Send message: ");
                System.out.println("2: Send message: ");
                input = in.readLine();
                sc1.sendMessage(input);
                sc2.sendMessage(input);
            }
        } catch (Exception ex) {
            System.err.println(ex);
        }
    }
    
    /**
     * thread that takes messages from output queue and sends them to the serial device
     */
    private class SendingThread extends Thread {
        OutputStream output;
        
        private SendingThread(OutputStream output) {
            this.output = output;
        }
        
        @Override
        public void run() {
            while (!isInterrupted()) {
                try {
                    String toSend = serialOutput.take();
                    byte[] message = toSend.getBytes(Charset.forName("UTF-8"));
                    output.write(message);
                    output.flush();
                } catch (IOException ex) {
                    serialInput.add("SendingThread: " + ex.toString());
                } catch (InterruptedException ex) {
                    interrupt();
                }
            }
        }
    }
}