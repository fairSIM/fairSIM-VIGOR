#
# fairSIM make file
#
# To work, either 'javac' has to point to a java 
# compiler (vers. 1.8), or change the line below to
# 'javac' instead of 'javac'

JC = javac
JAR = jar

# Options for the java compiler
EXTDIR="./external"

JFLAGS = -g  -Xlint:unchecked -Xlint:deprecation -extdirs ${EXTDIR} -d ./
#JFLAGS = -g -Xlint:unchecked -extdirs ${EXTDIR} -d ./
JFLAGS+= -target 1.8 -source 1.8


# remove command to clean up
RM = rm -vf

.PHONY:	all org/fairsim/git-version.txt

all:	
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


# misc rules
git-version :
	git rev-parse HEAD > org/fairsim/git-version.txt  ; \
	git tag --contains >> org/fairsim/git-version.txt ; \
	echo "n/a" >> org/fairsim/git-version.txt
	 	

jar:	git-version jtransforms-fork
	$(JAR) -cfm fairSIM_plugin_$(shell head -c 10 org/fairsim/git-version.txt).jar \
	Manifest.txt \
	org/fairsim/*/*.class \
	org/fairsim/extern/*/*.class \
	org/fairsim/git-version.txt \
	org/fairsim/resources/* \
	plugins.config \
	org/livesimextractor/*/*.class 

jar-wo-extern: git-version
	$(JAR) -cfm fairSIM_woJTransforms_plugin_$(shell head -c 10 org/fairsim/git-version.txt).jar \
	Manifest.txt \
	org/fairsim/*/*.class \
	org/fairsim/git-version.txt \
	org/fairsim/resources/* \
	plugins.config 


# shorthand for extracting the jtransforms-fork is necessary
jtransforms-fork: org/fairsim/extern/jtransforms/FloatFFT_3D.class

org/fairsim/extern/jtransforms/FloatFFT_3D.class:	
	$(JAR) -xvf external/jtransforms_fairSIM_fork.jar org/fairsim/extern/jtransforms 	

clean-jtransforms:
	$(RM) org/fairsim/external
	$(RM) org/fairsim/git-version-jtransforms.txt

# shorthand for generating the doc
doc:	doc/index.html

doc/index.html : $(wildcard org/fairsim/*/*.java) 
	javadoc -d doc/ -classpath ./ -extdirs ${EXTDIR} \
	-subpackages org.fairsim -exclude org.fairsim.extern.jtransforms 

clean : clean-jtransforms
	$(RM) fairSIM_*.jar fairSIM_*.tar.bz2
	$(RM) org/fairsim/*/*.class org/fairsim/git-version.txt
	$(RM) org/fairsim/extern/*/*.class
	$(RM) org/livesimextractor/*/*.class
	$(RM) -r doc/*
	$(RM) -r target

