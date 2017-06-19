#!/bin/bash

fileMissing=0

# Get the ImageJ base library (in version 1.49v, which is the lowest we support in the VIGOR-branch)
if [ ! -e ij149v.jar ] ; then
    fileMissing=1
    wget https://imagej.nih.gov/ij/download/jars/ij149v.jar
fi


# This fetches the java 1.6 runtime, needed for backwards-compatible
# compilation to Java 6 with newer compilers
#   unfortunately, Java 6's rt.jar does not seem to be on Maven, so
#   fetch it by extracting it from Ubuntu's openjdk deb.
#   Java 9 will make it much more easier with the "-release" option
if [ ! -e rt-1.6.jar ] ; then
    fileMissing=1

    mkdir tmp-rt-jar
    cd tmp-rt-jar

    # download the 'openjdk-6-jre-headless' deb file
    wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-6/openjdk-6-jre-headless_6b41-1.13.13-0ubuntu0.14.04.1_amd64.deb -O openjdk.deb


    # extract the deb
    echo "Extracting rt.jar from the .deb file"
    echo "This might take a few moments... "
    ar -x openjdk.deb

    # extract the rt.jar from the data.tar
    tar -xf data.tar.xz ./usr/lib/jvm/java-6-openjdk-amd64/jre/lib/rt.jar
    mv ./usr/lib/jvm/java-6-openjdk-amd64/jre/lib/rt.jar ../rt-1.6.jar
    cd ..

    rm -rf tmp-rt-jar
    echo "Done."

fi


# This fetches the serial comm library we use to talk to the Arduino
if [ ! -e nrjavaserial-3.12.0.jar ] ; then
    fileMissing=1
    wget https://github.com/NeuronRobotics/nrjavaserial/releases/download/3.12.0/nrjavaserial-3.12.0.jar
fi

# This extracts the MicroManager jars we need to compile the camera
# plugin. It requires '7z' from p7zip to be installed
if [ ! -e MMCoreJ.jar -o ! -e MMJ_.jar ] ; then
    fileMissing=1

    # checking for 7z to be available
    if [ "$(which 7z)" == ""  ] ; then
	echo "Please install '7z', e.g. by apt install p7zip-full"
	exit
    fi


    # downloading the micromanager install dmg (easier to handle than the exe)
    wget -c "http://valelab4.ucsf.edu/~MM/builds/1.4/Mac/Micro-Manager1.4.22.dmg"

    mkdir tmp-mm-jar
    cd tmp-mm-jar

    # extract the dmg file
    7z x ../Micro-Manager1.4.22.dmg
    
    # unpack the files from the image 
    7z x 2.hfs
    
    # 'MMCoreJ.jar' and 'MMJ_.jar' from that exe file
    cp Micro-Manager/Micro-Manager1.4/plugins/Micro-Manager/MMCoreJ.jar ../
    cp Micro-Manager/Micro-Manager1.4/plugins/Micro-Manager/MMJ_.jar ../

    # delete all the unused stuff
    cd ..
    rm -rf tmp-mm-jar

fi


# This fetches bUnwarpJ
if [ ! -e bUnwarpJ_-2.6.5.jar ] ; then
    fileMissing=1
    wget -c https://github.com/fiji/bUnwarpJ/releases/download/v2.6.5/bUnwarpJ_-2.6.5.jar
fi


# This fetches the ForthDD SLM API (only the Java part)
if [ ! -e forthDD-API.jar ] ; then
    fileMissing=1
    wget -c https://github.com/biophotonics-bielefeld/forthDD-API/releases/download/v1.0/forthDD-API.jar
fi



# Output a short feedback if all files are present
if [ $fileMissing -eq 0 ] ; then
    echo "All files found, fairSIM should compile"
fi

