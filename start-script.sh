#!/bin/bash

java -cp ./:./external/bUnwarpJ_-2.6.3.jar:./external/ij-1.50i.jar org.fairsim.livemode.LiveControlPanel vigor-fastsim-example.xml $1 $2 $3
