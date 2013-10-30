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
// This particular file is the main file for the model. It contains the 'main'
// entry function of that reads the input files, computes and outputs the reuse
// distance profile, and outputs modelled and measured cache miss rates.
//
// == File details
// Filename...........src/model/model.cpp
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...05-Sep-2013
//
//////////////////////////////////

// Include the header file
#include "model.h"

//////////////////////////////////
// Main entry function of the GPU cache model
//////////////////////////////////
int main(int argc, char** argv) {
	srand(time(0));
	std::cout << SPLIT_STRING << std::endl;
	message("");
	
	// Flush messages as soon as possible
	std::cout.setf(std::ios_base::unitbuf);
	
	// Read the hardware settings from file
	Settings hardware = get_settings();
	
	// Print cache statistics
	message("Cache configuration:");
	std::cout << "### \t Cache size: ~" << hardware.cache_bytes/1024 << "KB" << std::endl;
	std::cout << "### \t Line size: " << hardware.line_size << " bytes" << std::endl;
	std::cout << "### \t Layout: " << hardware.cache_ways << " ways, " << hardware.cache_sets << " sets" << std::endl;
	message("");
	
	// Parse the input argument and make sure that there is only one
	if (argc != 2) {
		message("Error: provide one argument only (a folder containing input trace files)");
		message("");
		std::cout << SPLIT_STRING << std::endl;
		exit(1);
	}
	std::string benchname = argv[1];
	
	// Loop over all found traces in the folder (one trace per kernel)
	for (unsigned kernel_id = 0; true; kernel_id++) {
		std::vector<Thread> threads(MAX_THREADS);
		for (unsigned t=0; t<MAX_THREADS; t++) { threads[t] = Thread(); }
		
		// Set the kernelname and include a counter
		std::string kernelname;
		if (kernel_id < 10) { kernelname = benchname+"_0"+std::to_string(kernel_id); }
		else {                kernelname = benchname+"_" +std::to_string(kernel_id); }
	
		// Load a memory access trace from a file
		Dim3 blockdim = read_file(threads, kernelname, benchname);
		unsigned blocksize = blockdim.x*blockdim.y*blockdim.z;
		
		// There was not a single trace that could be found - exit with an error
		if (blocksize == 0 && kernel_id == 0) {
			std::cout << "### Error: could not read file 'output/" << benchname << "/" << kernelname << ".trc'" << std::endl;
			message("");
			std::cout << SPLIT_STRING << std::endl;
			exit(1);
		}
		
		// The final tracefile is already processed, exit the loop
		if (blocksize == 0) { break; }
	
		// Assign threads to warps, threadblocks and GPU cores
		message("");
		std::cout << "### Assigning threads to warps/blocks/cores...";
		unsigned num_blocks = ceil(threads.size()/(float)(blocksize));
		unsigned num_warps_per_block = ceil(blocksize/(float)(hardware.warp_size));
		std::vector<std::vector<unsigned>> warps(num_warps_per_block*num_blocks);
		std::vector<std::vector<unsigned>> blocks(num_blocks);
		std::vector<std::vector<unsigned>> cores(hardware.num_cores);
		schedule_threads(threads, warps, blocks, cores, hardware, blocksize);
		std::cout << "done" << std::endl;
		
		// Model only a single core, modelling multiple cores requires a loop over 'cid'
		unsigned cid = 0;
		
		// Compute the number of active blocks on this core
		unsigned hardware_max_active_blocks = std::min(hardware.max_active_threads/blocksize, hardware.max_active_blocks);
		unsigned active_blocks = std::min((unsigned)cores[cid].size(), hardware_max_active_blocks);
		
		// Start the computation of the reuse distance profile
		message("");
		std::cout << "### [core " << cid << "]:" << std::endl;
		std::cout << "### Running " << active_blocks << " block(s) at a time" << std::endl;
		std::cout << "### Calculating the reuse distances";
		
		// Create a Gaussian distribution to model memory latencies
		std::random_device random;
		std::mt19937 gen(random());
		
		// Compute the reuse distance for 4 different cases
		std::vector<map_type<unsigned,unsigned>> distances(NUM_CASES);
		for (unsigned runs = 0; runs < NUM_CASES; runs++) {
			std::cout << "...";
			unsigned sets, ways;
			unsigned ml, ms, nml;
			unsigned mshr;
			
			// CASE 0 | Normal - full model
			sets = hardware.cache_sets; ways = hardware.cache_ways;
			ml = hardware.mem_latency; ms = hardware.mem_latency_stddev; nml = NON_MEM_LATENCY;
			mshr = hardware.num_mshr;
			
			// CASE 1 | Only 1 set: don't model associativity
			if (runs == 1) {
				sets = 1; ways = hardware.cache_ways*hardware.cache_sets;
			}
			
			// CASE 2 | Memory latency to 0: don't model latencies
			if (runs == 2) {
				ml = 0; ms = 0; nml = 0;
			}
			
			// CASE 3 | MSHR count to infinite: don't model MSHRs
			if (runs == 3) {
				mshr = INF;
			}
			
			// Calculate the reuse distance profile
			std::normal_distribution<> distribution(0,ms);
			reuse_distance(cores[cid], blocks, warps, threads, distances[runs], active_blocks, hardware,
			               sets, ways, ml, nml, mshr, gen, distribution);
		}
		std::cout << "done" << std::endl;
		
		// Process the reuse distance profile to obtain the cache hit/miss rate
		message("");
		output_miss_rate(distances, kernelname, benchname, hardware);
		
		// Display the cache hit/miss rate from the output of the verifier (if available)
		message("");
		verify_miss_rate(kernelname, benchname);
		message("");
	}
	
	// End of the program
	std::cout << SPLIT_STRING << std::endl;
	return 0;
}

//////////////////////////////////
