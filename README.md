# Introduction

This is a post-processing github repository to analyze a large number of HIT-SI simulations run using a single or two-temperature Hall-MHD model implemented in two MHD codes, NIMROD and PSI-Tet. This analysis will be published to show new physical understanding of HIT-SI plasmas with this more complex plasma model. 

Everything is in python and function/variable definitions are described in a doxygen file. To view the doxygen documentation, simply download the repo, cd into the directory, and type "doxygen Doxyfile" (requires the installation of doxygen on your machine). The html files can be viewed by opening a file like "html/index.html" in google chrome or safari. 

# Getting Started

Compatibility requires installation of Python 3.7, and the python packages scipy, numpy, matplotlib, and click. These packages can be installed through pip or through an anaconda interface. 

This code uses the "click" module for command line arguments. Much of the code is specific to HIT-SI format files from PSI-Tet and NIMROD. To see a list of the command line options, type "python HITSI.py --help". By default, the movies and pictures are written out to a directory called Pictures/. Create this directory or change the default in order to avoid errors. 

# Reproducing Results

All the .mat files for the analysis in the two-temperature paper are freely available upon request. Email akaptano@uw.edu for questions and requests. 

# License 

All files in this repository are available under the MIT License. Feel free to use and repurpose the code as you please.

The MIT License (MIT)

Copyright (c) 2019 Alan Kaptanoglu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

