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
// This particular file is implements 1) the mapping of threads to warps, thread-
// blocks and GPU cores, and 2) memory coalescing. The implementation of coalesc-
// ing is based on section "G.4.2. Global Memory" of the CUDA programming guide.
//
// == File details
// Filename...........src/model/scheduler.cpp
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...05-Sep-2013
//
//////////////////////////////////

// Include the header file
#include "model.h"

//////////////////////////////////
// Function to assign threads to warps/blocks/cores and to perform memory coalescing
//////////////////////////////////
void schedule_threads(std::vector<Thread> &threads,
                      std::vector<std::vector<unsigned>> &warps,
                      std::vector<std::vector<unsigned>> &blocks,
                      std::vector<std::vector<unsigned>> &cores,
                      const Settings hardware,
                      unsigned blocksize) {
	unsigned num_warps_per_block = ceil(blocksize/(float)(hardware.warp_size));
	
	// Assign threads to warps
	for (unsigned tid=0; tid<threads.size(); tid++) {
		unsigned wid = (tid%blocksize)/hardware.warp_size + (tid/blocksize)*num_warps_per_block;
		threads[tid].set_warp(wid);
		warps[wid].push_back(tid);
	}
	
	// Assign warps to threadblocks
	for (unsigned wnum=0; wnum<warps.size(); wnum++) {
		blocks[wnum/num_warps_per_block].push_back(wnum);
	}
	
	// Assign threadblocks to cores
	for (unsigned bnum=0; bnum<blocks.size(); bnum++) {
		cores[bnum%hardware.num_cores].push_back(bnum);
	}
	
	// Coalescing: iterate over all the warps
	for (unsigned wnum=0; wnum<warps.size(); wnum++) {
		
		// Coalescing: iterate over all the accesses for this warp
		unsigned done = 0;
		for (unsigned access=0; done<warps[wnum].size(); access++) {
		
			// Coalescing: iterate over all the threads in this warp
			for (unsigned tnum=0; tnum<warps[wnum].size(); tnum++) {
				unsigned tid = warps[wnum][tnum];
				
				// This thread has work to do
				if (access < threads[tid].accesses.size()) {
					
					// Compute the max schedule length (full-warps/half-warps/quarter-warps - see programming guide section "G.4.2. Global Memory")
					unsigned schedule_length;
					if      (threads[tid].accesses[access].bytes == 8)  { schedule_length = hardware.warp_size/2; }
					else if (threads[tid].accesses[access].bytes == 16) { schedule_length = hardware.warp_size/4; }
					else                                                { schedule_length = hardware.warp_size; }
					
					// See if the same cache-block has already been loaded for other threads in this warp
					unsigned long this_line = threads[tid].accesses[access].address/hardware.line_size;
					for (unsigned old_tnum=schedule_length*(tnum/schedule_length); old_tnum<tnum; old_tnum++) {
						unsigned old_tid = warps[wnum][old_tnum];
						unsigned long old_line = threads[old_tid].accesses[access].address/hardware.line_size;
						
						// The cache-block has been loaded earlier, coalescing the accesses
						if (this_line == old_line) {
							threads[tid].accesses[access].width = 0;
							if (threads[tid].accesses[access].address != threads[old_tid].accesses[access].address) {
								threads[old_tid].accesses[access].end_address = std::max(threads[old_tid].accesses[access].end_address, threads[tid].accesses[access].end_address);
								threads[old_tid].accesses[access].width++;
							}
							break;
						}
					}
				}
				
				// This thread is done
				else if (access == threads[tid].accesses.size()) {
					done++;
				}
			}
		}
	}
}

//////////////////////////////////
