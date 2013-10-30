#!/usr/bin/Rscript

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
## This file is an R visualisation script. It generates a histogram for each
## reuse distance profile found by the model. The only argument is the name of
## a benchmark.
##
## == File details
## Filename...........src/visualiser/histogram.r
## Author.............Cedric Nugteren <www.cedricnugteren.nl>
## Affiliation........Eindhoven University of Technology, The Netherlands
## Last modified on...05-Sep-2013
##
##################################

# Load the common commands/functions
source("src/visualiser/common.r")

##################################
## Constants
##################################

# Set the colours
colourset = c(blueish, redish)

##################################
## Start of the script
##################################

# Process the input arguments
options = commandArgs(trailingOnly=T)
dirname = d("output", options[1])

# Iterate over all output files
filenames = Sys.glob(p(d(dirname, options[1]), "*.out"))
for (filename in filenames) {

	# Configure the output plot
	outname = gsub(".out$", ".pdf", filename) 
	pdf(outname,height=4)

	# Parse an input file to collect data
	lines = parse(filename)

	# Loop over the data in the file
	values = c()
	frequencies = c()
	max = 0
	start = FALSE
	cache_ways = 1
	for (line in lines) {
		
		# Read the histogram data
		if (start) {
			items = as.integer(split(line, " "))
			if (length(items) == 2) {
				if (items[1] != 99999999) {
					values = c(values, items[1])
					frequencies = c(frequencies, items[2])
					if (items[1] > max) {
						max = items[1]
					}
				}
			}
			else {
				start = FALSE
			}
		}

		# Read the settings data
		if (!start) {
			items = split(line, " ")
			if (length(items) > 0) {
				if (items[1] == "histogram:") { start = T }
				if (items[1] == "cache_ways:") { cache_ways = as.integer(items[2]) }
			}
		}
	}

	# Set a maximum of the graph size to display
	max = cache_ways*4

	# Create the histogram from value/frequency data
	histogram = rep(0, max+1)
	if (length(values) > 0) {
		for (i in 1:length(values)) {
			if (values[i] < max) {
				histogram[values[i]+1] = frequencies[i]
			}
		}
	}

	# Plot the histogram
	distances = 0:max
	plot(distances, histogram, main=filename, type="b", xlab="reuse distance", ylab="frequency", lwd=2, col=colourset[1])
	abline(v=cache_ways-0.5, untf=F, lwd=2, col=colourset[2])
	
}

##################################