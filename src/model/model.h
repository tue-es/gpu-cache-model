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
// This particular file is the main header file for the model. It contains most
// of the pre-processor directives (includes/defines), several global settings,
// forward declarations of functions, and the definition of the following data-
// structures and classes:
// * Access...........struct
// * Dim3.............struct
// * Settings.........struct
// * Request..........struct
// * Thread...........class
// * Pool.............class
// * Requests.........class
//
// == File details
// Filename...........src/model/model.h
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...05-Sep-2013
//
//////////////////////////////////

#ifndef MODEL_H
#define MODEL_H

//////////////////////////////////
// Includes
//////////////////////////////////

// C++ headers
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <cmath>

// C headers
#include <assert.h>

// Custom includes
#include "tree.h"

//////////////////////////////////
// Unordered map (C++11) is better for performance, but a normal map also works
//////////////////////////////////
#if __cplusplus <= 199711L
	#warning Warning: please use a recent C++ standard (e.g. -std=c++0x) for better performance
	#define map_type std::map
#else
	#include <unordered_map>
	#define map_type std::unordered_map
#endif

//////////////////////////////////
// Settings
//////////////////////////////////
#define NUM_CORES 1             // Set the amount of cores (SMs) in the GPU
#define NON_MEM_LATENCY 0       // Set the latency of a cache hit
#define MAX_THREADS 32*1024     // Set the maximum number of threads supported

//////////////////////////////////
// Hardware properties
//////////////////////////////////
#define WARP_SIZE 32            // The size of a warp in threads
#define MAX_ACTIVE_THREADS 1536 // Maximum amount of threads active
#define MAX_ACTIVE_BLOCKS 8     // Maximum amount of threadblocks active

//////////////////////////////////
// IO defines
//////////////////////////////////
#define DISABLE_WARNINGS        // Disable or enable printing of warnings
#define WARNING_FACTOR 1.0      // Determine the threshold to print warnings
#define PRINT_MAX_DISTANCES 10  // Print only the X most interesting distances
#define SPLIT_STRING "###################################################"

//////////////////////////////////
// Other defines
//////////////////////////////////
#define INF 99999999            // Define infinite as a very large number
#define STACK_EXTRA_SIZE 256    // Extra size of the reuse distance stack
#define NUM_CASES 4             // Consider 4 cases: 1) normal, 2) full-associativity, 3) no latency, 4) infinite MSHRs

//////////////////////////////////
// Data-structure to describe a memory access
//////////////////////////////////
struct Access {
	unsigned direction;           // 1 for write, 0 for read
	unsigned long address;        // The byte address of the first byte
	unsigned width;               // The SIMD/coalescing width of the access
	unsigned bytes;               // The number of bytes accessed
	unsigned long end_address;    // The byte address of the last byte
};

//////////////////////////////////
// Data-structure for 2D or 3D items such as threadblocks or thread identifiers
//////////////////////////////////
struct Dim3 {
	unsigned x;
	unsigned y;
	unsigned z;
};

//////////////////////////////////
// Data-structure collecting all hardware settings
//////////////////////////////////
struct Settings {
	unsigned line_size;           // The size of a cache-line (in bytes)
	unsigned cache_bytes;         // Cache size (in bytes)
	unsigned cache_lines;         // Cache size (in lines)
	unsigned cache_ways;          // Number of ways or associativity (1 = direct mapped)
	unsigned cache_sets;          // Number of sets in a way (1 = fully associative)
	unsigned num_mshr;            // Number of miss-status hold registers (MSHRs)
	unsigned num_cores;           // Number of cores in the GPU (e.g. 14)
	unsigned warp_size;           // The size of a warp in threads (e.g. 32)
	unsigned max_active_threads;  // Maximum active threads in a core (e.g. 1536)
	unsigned max_active_blocks;   // Maximum active threadblocks in a core (e.g. 8)
	unsigned mem_latency;         // The best-case off-chip memory latency (e.g. 100)
	unsigned mem_latency_stddev;  // The standard deviation of the latency (e.g. 5)
};

//////////////////////////////////
// Data-structure to capture a memory request
//////////////////////////////////
struct Request {
	unsigned long addr;           // Memory address of the request
	unsigned set;                 // Set number of the request
};

//////////////////////////////////
// Class holding information about a GPU thread
//////////////////////////////////
class Thread {
	unsigned warpid;              // The thread's warp number
	unsigned blockid;             // The thread's block number

// Public variables and functions
public:
	unsigned pc;                  // The thread's 'program counter'
	std::vector<Access> accesses; // List of memory accesses to perform
	
	// Initialise the thread and set its program counter to zero
	Thread() {
		pc = 0;
		warpid = INF;
		blockid = INF;
	}
	
	// Add a new access to the list of accesses
	void append_access(Access access) {
		accesses.push_back(access);
	}
	
	// Take the next access and increment the program counter
	Access schedule() {
		pc++;
		assert(pc-1 < accesses.size());
		return accesses[pc-1];
	}
	
	// Put back the program counter: undo the previous schedule command
	void unschedule() {
		assert(pc > 0);
		pc--;
	}
	
	// Find out how many bytes the following access will have
	unsigned get_bytes() {
		if (pc == accesses.size()) { return 1; }
		else { return accesses[pc].bytes; }
	}
	
	// Find out if this thread has no more accesses to make
	bool is_done() {
		return (pc == accesses.size());
	}
	
	// Reset the program counter to zero
	void reset() {
		pc = 0;
	}
	
	// Set the thread's warp identifier to a given value
	void set_warp(unsigned _warpid) {
		assert(warpid == INF);
		warpid = _warpid;
	}
	
	// Set the thread's threadblock identifier to a given value
	void set_block(unsigned _blockid) {
		assert(blockid == INF);
		blockid = _blockid;
	}
};

//////////////////////////////////
// Class holding a pool of warps
//////////////////////////////////
class Pool {
	std::vector<unsigned> warps;           // A list of warps in the pool
	std::map<unsigned,unsigned> in_flight; // A list of in-flight warps
	unsigned size;                         // Size of the warp pool

// Public variables and functions
public:
	unsigned done;                         // Status implemented as a counter
	
	// Initialise the pool
	Pool() {
		done = 0;
		size = 0;
	}
	
	// Add a warp to the in-flight pool
	void add_warp(unsigned _warpid, unsigned future_time) {
		if (future_time == 0) {
			warps.push_back(_warpid);
		}
		else {
			in_flight[_warpid] = future_time;
		}
	}
	
	// Transfer warps from "in-flight" into the pool
	void process_warps_in_flight() {
		std::map<unsigned,unsigned>::iterator it = in_flight.begin();
		while (it != in_flight.end()) {
			if (it->second == 0) {
				warps.push_back(it->first);
				in_flight.erase(it++);
			}
			else {
				it->second--;
				++it;
			}
		}
	}
	
	// Take (and remove) a warp from the front (FIFO) of the pool
	unsigned take_warp() {
		unsigned warp = warps[0];
		warps.erase(warps.begin());
		return warp;
	}
	
	// Set the size of the pool
	void set_size() {
		size = warps.size();
	}
	
	// Find out if there is work in the pool at this time
	bool has_work() {
		return (warps.size() > 0);
	}
	
	// Find out whether all warps in the pool are done working
	bool is_done() {
		assert(size != 0);
		return (done == size);
	}
};

//////////////////////////////////
// Class containing outstanding memory requests
//////////////////////////////////
class Requests {
	std::map<unsigned,std::vector<Request>> request_list; // A list of outstanding requests
	std::set<unsigned> unique_requests;                   // List of unique outstanding requests

// Public variables and functions
public:
	
	// Initialise the pool of outstanding requests
	Requests() {
	}
	
	// Add a new request to the lists
	void add(unsigned long addr, unsigned future_time, unsigned set) {
		request_list[future_time].push_back(Request({addr,set}));
		unique_requests.insert(addr);
	}
	
	// Return the number of unique outstanding requests
	unsigned get_num_requests() {
		return unique_requests.size();
	}
	
	// Check whether there are current outstanding requests
	bool has_requests(unsigned current_time) {
		return (request_list[current_time].size() > 0);
	}
	
	// Process the current outstanding requests
	std::vector<Request> get_requests(unsigned current_time) {
		std::vector<Request> current = request_list[current_time];
		for (std::vector<Request>::iterator it = current.begin(); it != current.end(); it++) {
			Request request = *it;
			unique_requests.erase(request.addr);
		}
		request_list.erase(current_time);
		return current;
	}
};

//////////////////////////////////
// Forward declarations
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
                    std::normal_distribution<> distribution);
void process_requests(Requests &requests,
                      unsigned timestamp,
                      unsigned set,
                      map_type<unsigned long,unsigned> &P,
                      std::vector<Tree> &B,
                      std::vector<unsigned> &set_counters);
void schedule_threads(std::vector<Thread> &threads,
                      std::vector<std::vector<unsigned>> &warps,
                      std::vector<std::vector<unsigned>> &blocks,
                      std::vector<std::vector<unsigned>> &cores,
                      const Settings hardware,
                      unsigned block_size);
void output_miss_rate(std::vector<map_type<unsigned,unsigned>> &distances,
                      const std::string kernelname,
                      const std::string benchname,
                      const Settings hardware);
Dim3 read_file(std::vector<Thread> &threads,
               const std::string kernelname,
               const std::string benchname);
void verify_miss_rate(const std::string kernelname,
                      const std::string benchname);
unsigned line_addr_to_set(unsigned long line_addr,
                          unsigned long addr,
                          unsigned num_sets,
                          unsigned cache_bytes);
Settings get_settings(void);
void message(std::string x);

//////////////////////////////////

#endif