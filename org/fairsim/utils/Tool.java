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


/**
 * Logging and Timers
 * */
public final class Tool {

    /** Forbit the construction of this class */
    private Tool() {}
    /** The tool implementation in use*/
    static private Tool.Logger currentLogger;

    /** Simple logger */
    public interface Logger {
	public void writeTrace(String message); 
	public void writeError(String message, boolean fatal);
	public void writeShortMessage(String message);
    }

    /** Inits a standard tool */
    static {
	// basic logger goes to System.out
	currentLogger = new Tool.Logger() {
	    public void writeTrace(String message) {
		System.out.println( "[fairSIM] "+message);
		System.out.flush();
	    }
	    
	    public void writeError(String message, boolean fatal) {
		String prefix = (fatal)?("[fsFATAL]"):("[fsERROR]");
		System.err.println( prefix+" "+message);
		System.err.flush();
	    }
    
	    public void writeShortMessage(String message) {
		System.out.println( "-fairSIM- "+message);
		System.out.flush();
	    }
	};
	// want to catch exceptions
	/*
	Thread.setDefaultUncaughtExceptionHandler( new Thread.UncaughtExceptionHandler () {
	    public void uncaughtException(Thread t, Throwable e) {
		Tool.trace("Problem, caugt exception: "+e);
		e.printStackTrace();
	    }
	});
	*/
    }

    /** Write a trace message */
    static public final void trace(String message) {
	if (currentLogger!=null)
	    currentLogger.writeTrace( message );
    }
    
    /** Output a short status / info message */
    static public final void tell(String message) {
	if (currentLogger!=null)
	    currentLogger.writeShortMessage( message);
    }
   
    /** Write an error message */
    static public final void error(String message, boolean fatal ) {
	if (currentLogger!=null)
	    currentLogger.writeError( message, fatal );
    }


    /** Format a 'milliseconds since 1 Jan 1970' timestamp in ISO */ 
    static public String readableTimeStampMillis( long ms , boolean spaces ) {
	java.text.DateFormat df;
	if (!spaces) {
	    df = new java.text.SimpleDateFormat("yyyyMMdd'T'HHmmssZ");
	    df.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
	} else {
	    df = new java.text.SimpleDateFormat("yyyy-MM-dd' T 'HH:mm:ss '('Z')'");
	}

	String nowAsISO = df.format(new java.util.Date(ms));
	return nowAsISO;
    }

    /** Format a 'seconds since 1 Jan 1970' timestamp in ISO */ 
    static public String readableTimeStampSeconds( double seconds , boolean spaces) {
	long val = (long)(seconds*1000);
	return readableTimeStampMillis(val, spaces);
    }

    /** Implement and pass Tool.Logger to redirect log output,
     *  or set null to disable output completely */
    public static void setLogger( Tool.Logger t ) {
	currentLogger = t;
    }

    /** Shuts down all multi-threading pools */
    public static void shutdown() {
	SimpleMT.shutdown();
    }	

    /** Return a Tool.Timer, which is automatically started. */
    static public Timer getTimer() { return new Timer(); };
    
    /** A simple timer. TODO: The meaning of stop, pause, ... could
     *  be much clearer */
    public static class Timer {
	long start, stop, runtime, outtime;
	Timer() { 
	    start();
	}
	/** start the timer */
	public void start() { 
	    //start = System.currentTimeMillis(); 
	    start =  System.nanoTime(); 
	};
	/** stop the timer (next start resets it) */
	public void stop() { 
	    //stop = System.currentTimeMillis(); 
	    stop = System.nanoTime(); 
	    runtime += stop-start;
	    outtime=runtime;
	    runtime=0;
	    }
	/** pause the timer (next start continues) */
	public void hold(){
	    //stop = System.currentTimeMillis();
	    stop = System.nanoTime(); 
	    runtime += stop-start;
	    outtime  = runtime;
	    start =stop;
	}
	/** get the milliseconds on this timer */
	public double msElapsed() {
	    return outtime/1000000.;
	}

	/** output the amount of milliseconds counted */
	@Override public String toString(){ 
	    return String.format("%10.3f ms",(outtime/1000000.));
	}
    }


    /** A generic callback interface */
    public static interface Callback<T> {
	public void callback(T a);
    }

    /** A generic tuple */
    public static class Tuple<F,S> {
	public final F first;
	public final S second;

	public Tuple (F first, S second) {
	    this.first  = first;
	    this.second = second;
	}
    }


    /* TODO: compare this to utils.Future and such, and maybe finish it
    public static class Errant<D, Tool.Callback<R>> {
	
	final D val;
	final Tool.Callback<R> iface;

	protected Errant( D val, Tool.Callback<R> iface) {

	}


	public returnResult(R) {

    } */
     

}



