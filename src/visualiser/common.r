##################################
##
## == A reuse distance based GPU cache model
## This file is part of a cache model for GPUs. The cache model is based on
## reuse distance theory extended to work with GPUs. The cache model primarly
## focusses on modelling NVIDIA's Fermi architecture.
##
## == More information on the GPU cache model
## Article............A Detailed GPU Cache Model Based on Reuse Distance Theory
## Authors............Cedric Nugteren et al.
## Publication........Not published yet
##
## == Contents of this file
## This file is an R visualisation script. It includes the common constants and
## functions across all visualisation scripts.
##
## == File details
## Filename...........src/visualiser/common.r
## Author.............Cedric Nugteren <www.cedricnugteren.nl>
## Affiliation........Eindhoven University of Technology, The Netherlands
## Last modified on...05-Sep-2013
##
##################################

# Function to concatenate two strings
p <- function(a, b) { return(paste(a, b, sep="")) }
d <- function(a, b) { return(paste(a, b, sep="/")) }

# Function to split a string
split <- function(string, separator) { return(unlist(strsplit(string, separator))) }

# Function to open, read, and close a file
parse <- function(filename) {
	connection = file(filename,"rt")
	lines = readLines(connection)
	close(connection)
	return(lines)
}

# Set the colours
purplish = "#550077" # [ 85,  0,119] lumi=26
blueish  = "#4765b1" # [ 71,101,177] lumi=100
redish   = "#d67568" # [214,117,104] lumi=136
greenish = "#9bd4ca" # [155,212,202] lumi=199

# Set the gray tints
lgray    = "#969696" # [150,150,150] lumi=150
llgray   = "#c8c8c8" # [200,200,200] lumi=200
lllgray  = "#f2f2f2" # [242,242,242] lumi=242

##################################