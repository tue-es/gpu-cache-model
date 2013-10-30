//////////////////////////////////
//
// == A reuse distance based GPU cache model
// This file is part of a cache model for GPUs. The cache model is based on
// reuse distance theory extended to work with GPUs. The cache model primarly
// focusses on modelling NVIDIA's Fermi architecture.
//
// == More information on the GPU cache model
// Article............A Detailed GPU Cache Model Based on Reuse Distance Theory
// Authors............C. Nugteren et al.
//
// == Contents of this file
// This particular file is contains a function to determine how addresses are
// mapped to sets in a (hash) associative cache. In contains different settings,
// including a straightfoward non-hash mapping, a basic XOR hash mapping, and
// the Fermi GPU's hashing function.
//
// == File details
// Filename...........src/model/associativity.cpp
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...05-Sep-2013
//
//////////////////////////////////

// Include the header file
#include "model.h"

// Set the type of associativity address to set mapping to consider
#define MAPPING_TYPE 2

//////////////////////////////////
// Cache-line address to set mapping
//////////////////////////////////
unsigned line_addr_to_set(unsigned long line_addr,
                          unsigned long addr,
                          unsigned num_sets,
                          unsigned cache_bytes) {
	unsigned set = 0;
	
	// Generate groups of bits
	unsigned long addr_copy = line_addr;
	std::vector<unsigned> bits(32);
	for (unsigned i=0; i<bits.size(); i++) {
		if (addr_copy != 0) {
			bits[i] = addr_copy % 2;
			addr_copy = addr_copy / 2;
		}
		else {
			bits[i] = 0;
		}
	}
	
	// Default mapping function (no 'hash')
	if (MAPPING_TYPE == 0) {
		set = line_addr % num_sets;
	}
	
	// Basic XOR hashing function
	else if (MAPPING_TYPE == 1) {
		set = (line_addr % num_sets) ^ ((line_addr/num_sets) % num_sets);
	}
	
	// Fermi's hashing function
	else if (MAPPING_TYPE == 2) {
		unsigned b01234 = bits[0] + bits[1]*2 + bits[2]*4 + bits[3] *8 + bits[4] *16;
		unsigned b678AC = bits[6] + bits[7]*2 + bits[8]*4 + bits[10]*8 + bits[12]*16;
		assert(b01234 < 32);
		assert(b678AC < 32);
		set = (b01234 ^ b678AC) + bits[5]*32;
	}
	
	// Return the result modulo the number of sets
	return (set % num_sets);
}

//////////////////////////////////
