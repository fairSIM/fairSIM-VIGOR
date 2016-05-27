Sending camera images for live display. Protocol based on standard TCP streams,
each image split into header, followed by data. Simple protocoll:

ALL BYTE ORDER IS LITTLE ENDIAN!

## TCP header content (vers. 1)

byte	type	function
0	int64	sequence number 

8	int32	offset of payload (after header, curently has to be 0)
12	int32	length of payload (complete image, so width*height*bytesPerPixel)

16	int8	protocol version    (currently: 1)
17	int8	datatype of pixel data (currently: 2 - unsigned short)

18	uint8	"LIVESIM\0" (8 byte US-ASCII) 
26	6byte	currently not used	

32	int16	width of frame, in #pixel
34	int16	height of frame, in #pixel
36	int16	posA	for SIM: Position in angle
38	int16	posB	for SIM: Position in phase

40	int16	pos0	Position in z-plane
42	int16	pos1	Position in color (typically: use exitation wavelength)
44	int32	pos2	Position in time (integer, not stamp)

48	int64	cam-time    Timestamp (see below) camera
56	int64	cap-time    Timestamp (see below) capture computer
64	int64	rec-time    Timestamp (see below) receiving / storing computer

72	56 bytes    unused, for later upgrade

128	start payload, bytes: length*sizeof(datatype of pixel data)

## Timestamp

Timestamps are in integer microseconds since epoch.
Both "0x0000000000000000" as "0xFFffFFffFFffFFff" must be read as "timestamp not available".
Multiple timestamps are saved to avoid synchronization issues.

## Datatypes of pixel data (vers. 1)

0	NOT USED    ignore packet, e.g. TCP keep alive    
1	uint8	     8-bit 0..  255 range
2	uint16	    16-bit 0..65535 range


