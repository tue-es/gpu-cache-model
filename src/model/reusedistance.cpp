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
// This particular file implements an extended (GPU) version of Bennett and
// Kruskal's reuse distance theory algorithm as presented in literature (see
// below - figures 3-4, equations 1-2, section 4.4). The code keeps the names of
// the data-structures P and B as used in the article. It is based on a partial
// sum-hierarchy tree (see src/model/tree.h). It extends the original theory by
// modelling:
// 1) the GPU's hierarchy of threads/warps/blocks and sets of active threads
// 2) conditional and non-uniform (memory) latencies
// 3) cache associativity
// 4) miss-status holding-registers (MSHRs)
// 5) warp divergence (through a warp pool)
//
// == More information on reuse distance implementation
// Article............Calculating stack distances efficiently
// Authors............George Almasi, Calin Cascaval, and David Padua
// DOI................http://dx.doi.org/10.1145/773146.773043
//
// == File details
// Filename...........src/model/reusedistance.cpp
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...05-Sep-2013
//
//////////////////////////////////

// Include the header file
#include "model.h"

//////////////////////////////////
// Function to calculate the reuse distance for a single GPU core:
// * input: a vector of vectors containing the threads and their accesses
// * requires: the total amount of accesses to be able to construct the tree
// * output: a histogram (implemented as an unordered map) of the reuse distan-
//   ces (distance as key and frequency as value)
//////////////////////////////////
void reuse_distance(std::vector<unsigned> &core,
                    std::vector<std::vector<unsigned>> &blocks,
                    std::vector<std::vector<unsigned>> &warps,
                    std::vector<Thread> &threads,
                    map_type<unsigned,unsigned> &distances,
                    unsigned active_blocks,
                    const Settings hardware,
                    unsigned cache_sets,
                    unsigned cache_ways,
                    unsigned mem_latency,
                    unsigned non_mem_latency,
                    unsigned num_mshr,
                    std::mt19937 gen,
                    std::normal_distribution<> distribution) {
	
	// Prepare the computation of the number of accesses per set
	std::vector<unsigned> num_total_accesses(cache_sets);
	for (unsigned set=0; set<cache_sets; set++) {
		num_total_accesses[set] = 0;
	}
	
	// Compute the number of accesses per set (after coalescing has been performed)
	for (unsigned tid=0; tid<threads.size(); tid++) {
		while(!threads[tid].is_done()) {
			Access access = threads[tid].schedule();
			
			// Only consider accesses that haven't been disabled because of coalescing
			if (access.width != 0) {
				unsigned long line_addr = access.address/hardware.line_size;
				unsigned set = line_addr_to_set(line_addr,access.address,cache_sets,cache_sets*cache_ways*hardware.line_size);
				num_total_accesses[set]++;
				
				// Check if this access spans multiple cache-lines 
				unsigned long line_addr2 = access.end_address/hardware.line_size;
				if (line_addr != line_addr2) {
					set = line_addr_to_set(line_addr2,access.end_address,cache_sets,cache_sets*cache_ways*hardware.line_size);
					num_total_accesses[set]++;
				}
			}
		}
		
		// Reset the threads so that the PCs start at 0 again
		threads[tid].reset();
	}
	
	// Compute the grand total of accesses over all sets
	unsigned grand_total = 0;
	for (unsigned set=0; set<cache_sets; set++) {
		grand_total += num_total_accesses[set];
	}
	
	// Create a tree data structure for each set (B in the Almasi et al. paper)
	std::vector<Tree> B;
	B.reserve(cache_sets);
	for (unsigned set=0; set<cache_sets; set++) {
		B.emplace_back(num_total_accesses[set]+STACK_EXTRA_SIZE);
	}
	
	// Create the hash data structure (P in the Almasi et al. paper)
	map_type<unsigned long,unsigned> P;
	
	// Set the (fake) time to 0
	unsigned timestamp = 0;
	
	// Create the set-counters (starting at 1)
	std::vector<unsigned> set_counters(cache_sets);
	for (unsigned set=0; set<cache_sets; set++) {
		set_counters[set] = 1;
	}
	
	// Iterate round-robin over all the sets of active threads
	for (unsigned snum = 0; snum < ceil(core.size()/(float)(active_blocks)); snum++) {
		
		// Create the pool of warps and fill them with warps belonging to this set of active threads
		Pool pool = Pool();
		for (unsigned bnum = snum*active_blocks; bnum < (snum+1)*active_blocks && bnum < core.size(); bnum++) {
			unsigned bid = core[bnum];
			for (unsigned wnum = 0; wnum < blocks[bid].size(); wnum++) {
				pool.add_warp(blocks[bid][wnum],0);
			}
		}
		pool.set_size();
		
		// Create a pool of memory (misses) and non-memory (hits) requests
		std::vector<Requests> requests_miss(cache_sets);
		std::vector<Requests> requests_hit(cache_sets);
		
		// Loop over the warps in the warp pool
		while (!pool.is_done()) {
				
			// Check the status of the MSHRs
			unsigned num_miss_requests = 0;
			for (unsigned set = 0; set < cache_sets; set++) {
				num_miss_requests += requests_miss[set].get_num_requests();
			}
			
			// Check if there is currently work to do in the pool
			if (pool.has_work()) {
				
				// Select a warp from the pool
				unsigned wnum = pool.take_warp();
				unsigned max_future_time = 0;
				unsigned threads_done = 0;
				
				// Iterate over all the threads in this warp
				unsigned bytes = threads[warps[wnum][0]].get_bytes();
				unsigned portions = std::max(1u,bytes/4);
				for (unsigned warp_portion = 0; warp_portion < portions; warp_portion++) {
					unsigned tnum_start = warp_portion*(hardware.warp_size/portions);
					unsigned tnum_stop = (warp_portion+1)*(hardware.warp_size/portions);
				
					// Iterate as groups of warps/half-warps/quarter-warps depending on the access size (section G.4.2)
					for (unsigned tnum = tnum_start; tnum < tnum_stop && tnum < warps[wnum].size(); tnum++) {
						unsigned tid = warps[wnum][tnum];
						
						// Check if this thread is done now or still has work left
						if (threads[tid].is_done()) {
							threads_done++;
						}
						else {
							
							// Only schedule if the access is not performed by another thread (coalescing)
							Access access = threads[tid].schedule();
							if (access.width != 0) {
							
								// Compute the line address and the set
								unsigned long line_addr = access.address/hardware.line_size;
								unsigned set = line_addr_to_set(line_addr,access.address,cache_sets,cache_sets*cache_ways*hardware.line_size);
								assert(set < cache_sets);
								
								// Find the previous occurence
								unsigned previous_time = INF;
								if (P[line_addr]) {
									previous_time = P[line_addr];
									assert(previous_time < set_counters[set]);
								}
								
								// Find the reuse distance
								unsigned distance = INF;
								if (previous_time != INF) {
									distance = B[set].count(previous_time);
								}
								
								// Does not fit in the cache, mark as in-flight
								unsigned arrival_time;
								if (distance >= cache_ways) {
								
									// Compute the memory latency based on a half-normal distribution
									unsigned memory_latency = mem_latency + std::abs(std::round(distribution(gen)));
									arrival_time = timestamp + memory_latency;
									
									// Set this warp to return somewhere in the future
									if (memory_latency > max_future_time) {
										max_future_time = memory_latency;
									}
									
									// Check if there are no more free MSHRs for this request
									if (num_miss_requests >= num_mshr) {
										
										// Undo the changes made for this thread/warp and break
										if (tnum == 0) {
											threads[tid].unschedule();
											max_future_time = 0;
											break; // (breaks out of the loop over a warp)
										}
									}
									
									// Add the current request to the miss-request pool (with a delay)
									requests_miss[set].add(line_addr,arrival_time,set);
								}
								
								// ... does fit in the cache, assign a pipeline (hit) latency
								else {
									arrival_time = timestamp + non_mem_latency;
									
									// Add the current request to the hit-request pool (with a delay)
									requests_hit[set].add(line_addr,arrival_time,set);
								}
								
								// Store the reuse distance in a histogram
								if (!(distances[distance])) {
									distances[distance] = 0;
								}
								distances[distance]++;
							}
						}
					}
					
					// Process the earlier made requests (iterate over all sets)
					for (unsigned set = 0; set < cache_sets; set++) {
						process_requests(requests_hit[set],timestamp,set,P,B,set_counters);
						process_requests(requests_miss[set],timestamp,set,P,B,set_counters);
					}
				}
				
				// This warp is don: don't return it to the pool anymore
				if (threads_done == warps[wnum].size()) {
					pool.done++;
				}
				
				// Return the warp to the pool with a delay
				else {
					pool.add_warp(wnum,max_future_time);
				}
			}
			
			// Process the earlier made requests (iterate over all sets)
			for (unsigned set = 0; set < cache_sets; set++) {
				process_requests(requests_hit[set],timestamp,set,P,B,set_counters);
				process_requests(requests_miss[set],timestamp,set,P,B,set_counters);
			}
			
			// Process in-flight warps
			pool.process_warps_in_flight();
			
			// Increment the (fake) time
			timestamp++;
		}
	}
	
	// Reset all the program counters of the threads
	for (unsigned tid=0; tid<threads.size(); tid++) {
		threads[tid].reset();
	}
	
	// Sanity check to see if all accesses are made
	unsigned distances_total = 0;
	for(map_type<unsigned,unsigned>::iterator it=distances.begin(); it!= distances.end(); it++) {
		distances_total += it->second;
	}
	if (grand_total != distances_total) {
		std::cout << "Error: " << grand_total << " != " << distances_total << std::endl;
	}
}


//////////////////////////////////
// Function to process outstanding requests (actual modification of B and P)
//////////////////////////////////
void process_requests(Requests &requests,
                      unsigned timestamp,
                      unsigned set,
                      map_type<unsigned long,unsigned> &P,
                      std::vector<Tree> &B,
                      std::vector<unsigned> &set_counters) {
	if (requests.has_requests(timestamp)) {
		
		// Get all requests for the current time and handle them in-order
		std::vector<Request> current_requests = requests.get_requests(timestamp);
		for (unsigned r = 0; r < current_requests.size(); r++) {
			Request request = current_requests[r];
			
			// Find the previous occurence and remove it from the 'stack'
			unsigned previous_time = INF;
			if (P[request.addr]) {
				previous_time = P[request.addr];
				B[set].unset(previous_time);
			}
			
			// Set this time as the last used occurence
			P[request.addr] = set_counters[set];
			
			// Update the 'stack'
			B[set].set(set_counters[set]);
			set_counters[set]++;
		}
	}
}

//////////////////////////////////
