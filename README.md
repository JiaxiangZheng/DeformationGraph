DeformationGraph
================

A implemetation of deformation graph which was widely used in mesh deformation and non-rigid reconstruction. I write this for the reason that I will use it in later animation reconstruction work.

##about the code##
[Eigen](http://eigen.tuxfamily.org), [FLANN](http://www.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN) and [PCL](http://www.pointclouds.org/) are used as third party library.

`DeformGraph.h` and `DeformGraph.cpp` contains the core code of deformation
graph, and `DeformGraphTest.cpp` is an example for how to use the code. It
should be note that the initial data should be aligned together and normalized.


