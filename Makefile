#
# fairSIM make file
#
# To work, either 'java6' has to point to a java 
# compiler (vers. 1.6), or change the line below to
# 'java' instead of 'java6'

JC = javac6
JAR = jar

# Options for the java compiler
EXTDIR="./external"

JFLAGS = -g -Xlint:unchecked -Xlint:deprecation -extdirs ${EXTDIR} -d ./
#JFLAGS = -g -Xlint:unchecked -extdirs ${EXTDIR} -d ./
JFLAGS+= -target 1.6 -source 1.6


# remove command to clean up
RM = rm -vf

.PHONY:	all org/fairsim/git-version.txt

all:	jtrans
	$(JC) $(JFLAGS) org/fairsim/*/*.java

accelarator:
	$(JC) $(JFLAGS) org/fairsim/accel/*.java
fiji:
	$(JC) $(JFLAGS) org/fairsim/fiji/*.java
linalg:
	$(JC) $(JFLAGS) org/fairsim/linalg/*.java
livemode:
	$(JC) $(JFLAGS) org/fairsim/livemode/*.java
transport:
	$(JC) $(JFLAGS) org/fairsim/transport/*.java
sim_algorithm:
	$(JC) $(JFLAGS) org/fairsim/sim_algorithm/*.java
sim_gui:
	$(JC) $(JFLAGS) org/fairsim/sim_gui/*.java
tests:
	$(JC) $(JFLAGS) org/fairsim/tests/*.java
utils:
	$(JC) $(JFLAGS) org/fairsim/utils/*.java


jtrans:	org/fairsim/extern/jtransforms/FloatFFT_2D.class
org/fairsim/extern/jtransforms/FloatFFT_2D.class: $(wildcard org/fairsim/extern/jtransforms/*.java)
	$(JC) $(JFLAGS) org/fairsim/extern/jtransforms/*.java

# misc rules

git-version :
	git rev-parse HEAD > org/fairsim/git-version.txt  ; \
	git tag --contains >> org/fairsim/git-version.txt ; \
	echo "n/a" >> org/fairsim/git-version.txt
	 	

jarsrc	: git-verison
	$(JAR) -cvfm fairSIM-source_$(shell head -c 10 org/fairsim/git-version.txt).jar \
	Manifest.txt plugins.config \
	org/fairsim/git-version.txt \
	org/fairsim/*/*.class  org/fairsim/extern/*/*.class  \
	org/fairsim/resources/* \
	Makefile org/fairsim/*/*.java  org/fairsim/extern/*/*.java

tarsrc	: git-version
	tar -cvjf fairSIM-source_$(shell head -c 10 org/fairsim/git-version.txt).tar.bz2 \
	Manifest.txt plugins.config \
	org/fairsim/git-version.txt \
	org/fairsim/resources/* \
	Makefile org/fairsim/*/*.java  org/fairsim/extern/*/*.java
    

jar:	git-version	
	$(JAR) -cvfm fairSIM_plugin_$(shell head -c 10 org/fairsim/git-version.txt).jar \
	Manifest.txt plugins.config \
	org/fairsim/git-version.txt \
	org/fairsim/resources/* \
	org/fairsim/*/*.class  org/fairsim/extern/*/*.class 


doc:	doc/index.html

doc/index.html : $(wildcard org/fairsim/*/*.java) 
	javadoc -d doc/ -classpath ./ -extdirs ${EXTDIR} \
	-subpackages org.fairsim -exclude org.fairsim.extern.jtransforms 
#	org/fairsim/*/*.java

clean :
	$(RM) fairSIM_*.jar fairSIM_*.tar.bz2
	$(RM) org/fairsim/*/*.class org/fairsim/git-version.txt
	$(RM) -r doc/*

clean-all: clean
	$(RM) org/fairsim/extern/*/*.class

