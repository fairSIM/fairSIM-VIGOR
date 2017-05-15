/*
 * Code to communicate with the arduino.
 * This is a modifyed class of the "Sample Java Code" from "playground.arduino.cc"
 * http://playground.arduino.cc/Interfacing/Java
 * Original code under CC-BY-SA-3.0
 */
package org.fairsim.utils;

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
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Class to control the arduino
 *
 * @author m.lachetta
 */
public class ArduinoCommunicator {

    private SerialPort serialPort;
    private final BlockingQueue<String> serialInput;
    private final BlockingQueue<String> arduinoCommands;
    private SendingThread sendingThread;
    /**
     * The port we're normally going to use.
     */
    private static final String PORT_NAMES[] = {
        "/dev/tty.usbserial-A9007UX1", // Mac OS X
        "/dev/ttyACM0", // Raspberry Pi
        "/dev/ttyUSB0", // Linux
        "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", // Windows
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
    private final int DATA_RATE;
    
    /**
     * constructor
     */
    ArduinoCommunicator(int dataRate) {
        DATA_RATE = dataRate;
        serialInput = new LinkedBlockingQueue<>();
        arduinoCommands = new LinkedBlockingQueue<>();
    }

    /**
     * initializes the serial connection to the arduino
     * @throws Exception 
     */
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
            throw new NullPointerException(massage);
        }

        // open serial port, and use class name for the appName.
        serialPort = (SerialPort) portId.open(this.getClass().getName(), TIME_OUT);

        // set port parameters
        serialPort.setSerialPortParams(DATA_RATE, SerialPort.DATABITS_8, SerialPort.STOPBITS_1, SerialPort.PARITY_NONE);

        // open the streams
        input = new BufferedReader(new InputStreamReader(serialPort.getInputStream()));
        output = serialPort.getOutputStream();

        // add event listeners
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
        serialPort.addEventListener(serialListener);
        serialPort.notifyOnDataAvailable(true);
    }

    /**
     * This should be called when you stop using the port. This will prevent
     * port locking on platforms like Linux.
     */
    private synchronized void close() {
        if (serialPort != null) {
            serialPort.removeEventListener();
            serialPort.close();
        }
    }
    
    
    /**
     * 
     * @param timeout in milli seconds
     * @return the next message from the arduino or null if timed out
     * @throws InterruptedException if the poll command gets interrupted
     * @throws IOException if a communication error occurs
     */
    public String pollSerialInput(int timeout) throws InterruptedException, IOException {
        String input = serialInput.poll(timeout, TimeUnit.MILLISECONDS);
        return handleMessage(input);
    }
    
    /**
     * 
     * @return the next message from the arduino or null if there is no message
     * @throws IOException if a communication error occurs
     */
    public String pollSerialInput() throws IOException {
        String input = serialInput.poll();
        return handleMessage(input);
    }
    
    /**
     * 
     * @return the next message from the arduino or blocks until there is a massage
     * @throws InterruptedException if the take command gets interrupted
     * @throws IOException IOException if a communication error occurs
     */
    public String takeSerialInput() throws InterruptedException, IOException {
        String input = serialInput.take();
        return handleMessage(input);
    }
    
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
     * builds up a connection to the arduino
     */
    public void connect() throws IOException {
        try {
            initialize();
            sendingThread = new SendingThread();
            sendingThread.start();
        } catch (Exception e) {
            throw new IOException(e.getMessage());
        }
    }

    /**
     * closes the connection to the arduino
     */
    public void disconnect() {
        try {
            if (sendingThread != null) sendingThread.interrupt();
            arduinoCommands.clear();
            serialInput.clear();
            if (sendingThread != null) sendingThread.join();
            close();
        } catch (InterruptedException e) {
            throw new RuntimeException("Do not interrupt this");
        }
    }
    
    /**
     * sends a command to the arduino
     * @param command command for the arduino
     */
    public void sendCommand(String command) {
        arduinoCommands.add(command);
    }

    /**
     * main method for testing direct serial communication with the arduino
     */
    public static void main(String[] args) throws IOException, InterruptedException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String input;
        try {
            ArduinoCommunicator main = new ArduinoCommunicator(19200);
            main.connect();
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        while (true) {
                            System.out.println("Arduino: " + main.takeSerialInput() );
                        }
                    } catch (Exception ex) {
                        System.err.println(ex);
                    }
                }
            }).start();
            while (true) {
                System.out.println("Send command: ");
                input = in.readLine();
                main.sendCommand(input);
            }
        } catch (Exception ex) {
            System.err.println(ex);
        }
    }
    
    /**
     * thread that takes command from commad queue and sends them to the arduino
     */
    private class SendingThread extends Thread {
        @Override
        public void run() {
            while (!isInterrupted()) {
                try {
                    String toSend = arduinoCommands.take();
                    byte[] command = toSend.getBytes(Charset.forName("UTF-8"));
                    output.write(command);
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

