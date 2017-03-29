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
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import org.fairsim.utils.Tool;

/**
 * Class to control the arduino
 *
 * @author m.lachetta
 */
public class ArduinoController implements SerialPortEventListener {

    private SerialPort serialPort;
    private final ControllerServerGui serverGui;
    private final BlockingQueue<String> arduinoAnswers;
    private final BlockingQueue<String> arduinoCommands;
    private SendingThread sendingThread;
    private boolean ready; // is arduino ready for a command?
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
    private static final int DATA_RATE = 19200;

    /**
     * constructor for testing serial communication in command line
     * @throws Exception 
     */
    private ArduinoController() throws Exception {
        serverGui = null;
        arduinoAnswers = null;
        arduinoCommands = null;
        initialize();
    }
    
    /**
     * constructor
     * @param serverGui gui of this
     */
    ArduinoController(ControllerServerGui serverGui) {
        this.serverGui = serverGui;
        arduinoAnswers = new LinkedBlockingQueue<>();
        arduinoCommands = new LinkedBlockingQueue<>();
        ready = false;
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
            try {
                serverGui.showText("Arduino: " + massage);
            } catch (NullPointerException ex) {
                System.out.println("Arduino: " + massage);
            }
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
        serialPort.addEventListener(this);
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
     * Handle an event on the serial port. Adds the arduinos answer to the
     * answers queue
     */
    @Override
    public synchronized void serialEvent(SerialPortEvent oEvent) {
        if (oEvent.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
            try {
                String inputLine = input.readLine();
                if (inputLine.equals("ready")) ready = true; // will not be added to the answers queue
                else {
                    if (inputLine.startsWith("Movie started: ")) ready = true;
                    try {
                        // for fairSIM mode
                        serverGui.showText("Arduino: " + inputLine);
                        arduinoAnswers.add(inputLine);
                    } catch (NullPointerException ex) {
                        // for testing mode
                        System.out.println("Arduino: " + inputLine);
                    }
                }
            } catch (Exception e) {
                // ecception handling
                if (e instanceof IOException && e.getMessage().equals("Underlying input stream returned zero bytes")) {
                    //happens sometimes but seams not to be important, will be ignorred
                } else {
                    // outputs an error
                    System.err.println("Arduino: Error:" + e.toString());
                    try {
                        serverGui.showText("Arduino: Error: " + e.toString());
                    } catch (NullPointerException ex) {
                    }
                }
            }
        }
        // Ignore all the other eventTypes, but you should consider the other ones.
    }
    
    /**
     * asks the arduino for the list of running orders and returns them as an
     * encoded array
     * @return encoded array of running orders from the arduino
     */
    public String getRoList() {
        String error = "Error: ArduinoController.getRoList()";
        arduinoCommands.add("list");
        String firstAnswer = getArduinoAnswer();
        if (!firstAnswer.startsWith("list:start:")) return error;
        try {
            int roMax = Integer.parseInt(firstAnswer.split(":")[2]);
            String[] runningOrders = new String[roMax];
            for (int i = 0; i < roMax; i++) {
                runningOrders[i] = getArduinoAnswer();
            }
            String lastAnswer = getArduinoAnswer();
            if (!lastAnswer.equals("list:end")) return error;
            return Tool.encodeArray("Transfering rolist", runningOrders);
        } catch (NumberFormatException ex) {
            return error;
        }
    }
    
    /**
     * @return the next answer from the arduino or a timeout message after 1 sec
     */
    private String getArduinoAnswer() {
        try {
            String answer = arduinoAnswers.poll(1000, TimeUnit.MILLISECONDS);
            if (answer != null) return answer;
            else return "Error: No answer from arduino";
        } catch (InterruptedException ex) {
            return "Error: " + ex.toString();
        }
    }
    
    /**
     * sends a command to the arduino, only for testing from commandline
     * @param toSend command for the arduino
     * @throws IOException if communication with arduino went wrong
     */
    private void sendCommand(String toSend) throws IOException {
        byte[] command = toSend.getBytes(Charset.forName("UTF-8"));
        output.write(command);
        output.flush();
    }
    
    /**
     * builds up a connection to the arduino and starts the sending thread
     * @return information about connecting status
     */
    public String connect() {
        try {
            initialize();
            sendingThread = new SendingThread();
            sendingThread.start();
            return "Connected to the arduino";
        } catch (Exception e) {
            return "Error: " + e.toString();
        }
    }

    /**
     * closes the connection to the arduino
     * @return information about connecting status
     */
    public String disconnect() {
        try {
            if (sendingThread != null) sendingThread.interrupt();
            arduinoCommands.clear();
            arduinoAnswers.clear();
            sendingThread.join();
            close();
            return "Disconnected from the arduino";
        } catch (InterruptedException e) {
            return "Error: " + e.toString();
        }
    }
    
    /**
     * sends a single char to the arduino
     * @param c char to be send
     * @return answer of the arduino
     */
    public String sendChar(char c) {
        arduinoCommands.add(String.valueOf(c));
        return getArduinoAnswer();
    }
    
    /**
     * gives the arduino the command to take a photo with the fast sim setup
     * @param runningOrder running order for the photo
     * @return answer for the arduino
     */
    public String takePhoto(int runningOrder) {
        arduinoCommands.add("p;" + runningOrder);
        return getArduinoAnswer();
    }
    
    /**
     * gives the arduino the command to start movie acquisition with the fast sim setup
     * @param runningOrder running order for the acquisition
     * @param breakTime delays between sim sequences
     * @return answer from the arduino
     */
    public String startMovie(int runningOrder, int breakTime) {
        arduinoCommands.add("m;" + runningOrder + ";" + breakTime);
        return getArduinoAnswer();
    }

    /**
     * main method for testing direct serial communication with the arduino
     * @param args not used
     * @throws IOException if anything went wrong
     * @throws InterruptedException if anything went wrong
     */
    public static void main(String[] args) throws IOException, InterruptedException {
        try {
            ArduinoController main = new ArduinoController(null);
            main.connect();
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            String input;
            while (true) {
                System.out.println("Command for the Microcontroller: ");
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
                    if (ready) {
                        ready = false;
                        String toSend = arduinoCommands.take();
                        byte[] command = toSend.getBytes(Charset.forName("UTF-8"));
                        output.write(command);
                        output.flush();
                    } else {
                        sleep(100);
                    }
                } catch (IOException ex) {
                    arduinoAnswers.add("SendingThread: " + ex.toString());
                } catch (InterruptedException ex) {
                    interrupt();
                    continue;
                }
            }
        }
    }
}

