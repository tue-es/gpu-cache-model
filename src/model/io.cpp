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
// This particular file implements all the input/output functions of the cache
// model. This includes input/output to disk (files) and input/output to stdout.
//
// == File details
// Filename...........src/model/io.cpp
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...05-Sep-2013
//
//////////////////////////////////

// Include the header file
#include "model.h"

//////////////////////////////////
// Global settings for the directory structure of the GPU cache model
//////////////////////////////////
std::string output_dir = "output";
std::string temp_dir = "temp";
std::string config_dir = "configurations";

//////////////////////////////////
// Function to parse the memory access trace (input)
//////////////////////////////////
Dim3 read_file(std::vector<Thread> &threads,
               const std::string kernelname,
               const std::string benchname) {
	unsigned num_threads = 0;
	unsigned num_accesses = 0;
	std::string filename = output_dir+"/"+benchname+"/"+kernelname+".trc";
	
	// Test if the file exists, return if it does not exist
	std::ifstream exists_file(filename);
	if (!exists_file) {
		return Dim3({0,0,0});
	}
	
	// Open the file for reading
	std::cout << SPLIT_STRING << std::endl;
	message("");
	std::cout << "### Reading the trace file for '" << kernelname << "'...";
	std::ifstream input_file(filename);
	
	// First get the blocksize from the trace file
	std::string temp_string;
	Dim3 blockdim;
	input_file >> temp_string >> blockdim.x >> blockdim.y >> blockdim.z;
	
	// Then proceed to the actual trace data
	unsigned thread, direction, bytes;
	unsigned long address;
	while (input_file >> thread >> direction >> address >> bytes) {
		
		// Consider only loads (stores are not cached in Fermi's L1 caches)
		if (direction == 0) {
		
			// Count the number of accesses and threads
			num_accesses++;
			num_threads = (num_threads > thread) ? num_threads : thread + 1;
			
			// Store the data in the Thread class
			Access access = { direction, address, 1, bytes, address+bytes-1 };
			threads[thread].append_access(access);
		}
	}
	std::cout << "done" << std::endl;
	
	// Test if the file actually contained memory accesses - exit otherwise
	if (!(num_accesses > 0 && num_threads > 0)) {
		std::cout << "### Error: '" << filename << "' is not a valid memory access trace" << std::endl;
		message("");
		return Dim3({0,0,0});
	}
	
	// Reduce the size of the threads vector
	threads.resize(num_threads);
	threads.shrink_to_fit();
	
	// Print additional information and return the threadblock dimensions
	std::cout << "### Blocksize: (" << blockdim.x << "," << blockdim.y << "," << blockdim.z << ")" << std::endl;
	std::cout << "### Total threads: " << num_threads << std::endl;
	std::cout << "### Total memory accesses: " << num_accesses << "" << std::endl;
	return blockdim;
}

//////////////////////////////////
// Function to output the histogram and the cache miss rate to file and stdout
//////////////////////////////////
void output_miss_rate(std::vector<map_type<unsigned,unsigned>> &distances,
                      const std::string kernelname,
                      const std::string benchname,
                      const Settings hardware) {
	
	// Prepare the output file and output some hardware settings
	std::ofstream file;
	file.open(output_dir+"/"+benchname+"/"+kernelname+".out");
	file << "line_size: " << hardware.line_size << std::endl;
	file << "cache_bytes: " << hardware.cache_bytes << std::endl;
	file << "cache_lines: " << hardware.cache_lines << std::endl;
	file << "cache_ways: " << hardware.cache_ways << std::endl;
	file << "cache_sets: " << hardware.cache_sets << std::endl;
	
	// Sort the reuse distances and print them to file
	std::map<unsigned,unsigned> sorted_distances;
	file  << std::endl << "histogram:" << std::endl;
	for(map_type<unsigned,unsigned>::iterator it=distances[0].begin(); it!= distances[0].end(); it++) {
		file << "" << it->first << " " << it->second << std::endl;
		sorted_distances.insert(std::make_pair((unsigned)it->second,(unsigned)it->first));
	}
	file << std::endl;
	
	// Print the sorted reuse distance histogram to stdout
	message("Printing results as [reuse_distance] => frequency: ");
	unsigned count = 0;
	for(std::map<unsigned,unsigned>::reverse_iterator it=sorted_distances.rbegin(); it!= sorted_distances.rend(); it++) {
		
		// Print to stdout
		if (it->second == INF) { std::cout << "### %%% [inf] => " << it->first << "" << std::endl; }
		else { std::cout << "### %%% [" << it->second << "] => " << it->first << "" << std::endl; }
		
		// Break after printing the X most interesting values
		if (count > PRINT_MAX_DISTANCES) { break; }
		count++;
	}
	
	// Prepare to gather the cache miss rates
	message("");
	std::cout << "### Modeled cache miss rate:" << std::endl;
	unsigned miss_compulsory[NUM_CASES] = {0, 0, 0, 0};
	unsigned miss_capacity[NUM_CASES] = {0, 0, 0, 0};
	unsigned miss[NUM_CASES];
	unsigned hits = 0;
	
	// Compute the cache misses for the 4 different cases
	for (int i=0; i<NUM_CASES; i++) {
		for(map_type<unsigned,unsigned>::iterator it=distances[i].begin(); it!= distances[i].end(); it++) {
			unsigned cache_ways = hardware.cache_ways;
			if (i == 1) { cache_ways = hardware.cache_ways*hardware.cache_sets; }
			
			// Compute the compulsory and capacity misses
			if (it->first == INF) {
				miss_compulsory[i] += it->second;
			}
			else if (it->first > cache_ways ) {
				miss_capacity[i] += it->second;
			}
			
			// Compute the hits
			else if (i == 0) {
				hits += it->second;
			}
		}
		miss[i] = miss_compulsory[i] + miss_capacity[i];
	}
	
	// Check for possible problems
	#ifdef ENABLE_WARNINGS
		if ((float)miss[1] > (float)miss[0]*WARNING_FACTOR) {
			std::cout << "### [warning] more misses with full-associativity (" << miss[1] << ") than with set-associativity (" << miss[0] << ")" << std::endl;
		}
		if ((float)miss[2] > (float)miss[0]*WARNING_FACTOR) {
			std::cout << "### [warning] more misses without latency (" << miss[2] << ") than with latency (" << miss[0] << ")" << std::endl;
		}
		if ((float)miss[3] > (float)miss[0]*WARNING_FACTOR) {
			std::cout << "### [warning] more misses with unlimited MSHRs (" << miss[3] << ") than with limited MSHRs (" << miss[0] << ")" << std::endl;
		}
	#endif
	
	// Compute the various types of cache miss rates
	int miss_associativity = miss[0] - miss[1];
	int miss_latency       = miss_compulsory[0] - miss_compulsory[2];
	int miss_mshr          = miss[0] - miss[3];
	miss_compulsory[0] = miss_compulsory[2];
	int rest = miss[0] - (miss_compulsory[0] + std::max(0,miss_latency) + std::max(0,miss_associativity) + std::max(0,miss_mshr));
	miss_capacity[0] = std::max(0,rest);
	if (rest < 0) {
		if (miss_mshr > -rest) {         miss_mshr          = miss_mshr          - rest; }
		else if (miss_latency > -rest) { miss_latency       = miss_latency       - rest; }
		else {                           miss_associativity = miss_associativity - rest; }
	}
	
	// Compute the final cache miss rates
	unsigned total_misses = miss[0];
	unsigned total_accesses = total_misses + hits;
	float miss_rate = 100*total_misses/(float)(total_accesses);

	// Report the cache hit/miss rates to stdout
	std::cout << "### \t Total accesses: "         << total_accesses << std::endl;
	std::cout << "### \t Of which are misses: "    << miss_compulsory[0] << " + " << miss_capacity[0] << " + " << std::max(0,miss_associativity) << " + " << std::max(0,miss_latency) << " + " << std::max(0,miss_mshr) << " = " << total_misses << " (compulsory + capacity + associativity + latency + mshr = total)" << std::endl;
	std::cout << "### \t Of which are hits: "      << hits << std::endl;
	std::cout << "### \t Miss rate: "              << miss_rate << "%" << std::endl;
	
	// Report the cache hit/miss rates to file
	file << "modelled_accesses: "                  << total_accesses                  << std::endl;
	file << "modelled_misses(compulsory): "        << miss_compulsory[0]              << std::endl;
	file << "modelled_misses(capacity): "          << miss_capacity[0]                << std::endl;
	file << "modelled_misses(associativity): "     << std::max(0,miss_associativity)  << std::endl;
	file << "modelled_misses(latency): "           << std::max(0,miss_latency)        << std::endl;
	file << "modelled_misses(mshr): "              << std::max(0,miss_mshr)           << std::endl;
	file << "modelled_misses(tot_associativity): " << miss[1]                         << std::endl;
	file << "modelled_misses(tot_latency): "       << miss[2]                         << std::endl;
	file << "modelled_misses(tot_mshr): "          << miss[3]                         << std::endl;
	file << "modelled_hits: "                      << hits                            << std::endl;
	file << "modelled_miss_rate: "                 << miss_rate                       << std::endl;
	
	// Close the output file
	file.close();
}

//////////////////////////////////
// Read the verifier output (from hardware execution) and display the results
//////////////////////////////////
void verify_miss_rate(const std::string kernelname,
                      const std::string benchname) {
	std::string filename = output_dir+"/"+benchname+"/"+kernelname+".prof";
	
	// Test if the file exists
	std::ifstream exists_file(filename);
	if (!exists_file) {
		message("No verifier data information available, skipping verification");
		return;
	}
	
	// Parse the file line by line
	unsigned counter = 0;
	unsigned long data;
	unsigned long miss = 0;
	unsigned long hit = 0;
	std::ifstream input_file(filename);
	while (input_file >> data) {
		if (counter == 0) { hit = data; }
		if (counter == 1) { miss = data; }
		counter++;
	}
	
	// Prepare to append to the output file
	std::ofstream file;
	file.open(output_dir+"/"+benchname+"/"+kernelname+".out", std::fstream::app);
	file << std::endl;
	
	// Output verification data to stdout
	message("Cache miss rate according to verification data:");
	float miss_rate = 100*miss/(double)(miss+hit);
	std::cout << "### \t Total accesses: " << (miss+hit) << std::endl;
	std::cout << "### \t Misses: " << miss << std::endl;
	std::cout << "### \t Hits: " << hit << std::endl;
	std::cout << "### \t Miss rate: " << miss_rate << "%" << std::endl;
	
	// Output verification data to file
	file << "verified_misses: " << miss << std::endl;
	file << "verified_hits: " << hit << std::endl;
	file << "verified_miss_rate: " << miss_rate << std::endl;
	
	// Close the output file
	file.close();
}

//////////////////////////////////
// Function to read the hardware settings from a file
//////////////////////////////////
Settings get_settings(void) {
	std::string filename = config_dir+"/"+"current.conf";
	
	// Test if the file exists
	std::ifstream exists_file(filename);
	if (!exists_file) {
		std::cout << "### Error: could not read settings file '" << filename << "'" << std::endl;
		message("");
		exit(0);
	}
	
	// Open the settings file for reading
	std::ifstream input_file(filename);
	
	// Then proceed to the parse the data
	std::string identifier;
	unsigned line_size, cache_bytes, cache_ways, num_mshr, mem_latency, mem_latency_stddev;
	input_file >> identifier >> line_size;
	input_file >> identifier >> cache_bytes;
	input_file >> identifier >> cache_ways;
	input_file >> identifier >> num_mshr;
	input_file >> identifier >> mem_latency;
	input_file >> identifier >> mem_latency_stddev;
	
	// Store the data in the settings data-structure
	Settings hardware = {
	  line_size,
	  cache_bytes,
	  cache_bytes/line_size,
	  cache_ways,
	  cache_bytes/(line_size*cache_ways),
	  num_mshr,
	  NUM_CORES,
	  WARP_SIZE,
	  MAX_ACTIVE_THREADS,
	  MAX_ACTIVE_BLOCKS,
	  mem_latency,
	  mem_latency_stddev
	};
	
	// Close the file and return
	return hardware;
}

//////////////////////////////////
// Helper function to print messages to stdout
//////////////////////////////////
void message(std::string x) {
	std::cout << "### " << x << std::endl;
}

//////////////////////////////////
