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
// This file provides the Ocelot-based tracer. The tracer takes as input a CUDA
// program emulated in Ocelot and outputs all memory accesses made per thread
// (not in the real execution order - it is just an emulation). The output is
// written to a file and can be limited to a certain amount of threads.
//
// == File details
// Filename...........src/tracer/tracer.cpp
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...30-Oct-2013
//
//////////////////////////////////

// Ocelot includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/trace/interface/TraceEvent.h>
#include <ocelot/executive/interface/ExecutableKernel.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/ir/interface/PTXOperand.h>

// C++ includes
#include <iostream>
#include <fstream>
#include <map>

// Set the maximum amount of threads 
#define MAX_THREADS (8*1024)

//////////////////////////////////
// The trace generator class
//////////////////////////////////
class TraceGenerator : public trace::TraceGenerator {
	
	// Counters
	unsigned long loadCounter;
	unsigned long storeCounter;
	unsigned long computeCounter;
	unsigned long memoryCounter;
	unsigned kernel_id;
	unsigned threads;
	
	// Status
	bool finished;
	bool initialised;
	
	// Base address
	unsigned long baseAddress;
	
	// Mapping of CUDA thread IDs (gid) to trace thread IDs (tid)
	std::map<unsigned,unsigned> gids;
	
	// File streams
	std::ofstream addrFile;
	
	// Name of the program
	std::string name;
	
	// Public methods
	public:
	
	// Constructor with filename
	TraceGenerator(std::string _name="default") {
		name = _name;
		kernel_id = 0;
	}
	
	// Close output files
	void finish() {
		if (!finished) {
			finalise();
		}
		float ratio = computeCounter/(float)memoryCounter;
		std::cout << "[Finished kernel] Loads: " << loadCounter << ", Stores: " << storeCounter << "\n";
		std::cout << "[Finished kernel] Compute (" << computeCounter << ") memory (" << memoryCounter << ") ratio: " << ratio << "\n";
		//abort(); // End Ocelot. The exit(1) function does not seem to work.
	}
	
	// Open output files
	void initialize(const executive::ExecutableKernel & kernel) {
		std::cout << "Starting with " << kernel.name << "" << std::endl;
		loadCounter = 0;
		storeCounter = 0;
		computeCounter = 0;
		memoryCounter = 0;
		baseAddress = 0;
		threads = 0;
		finished = false;
		initialised = false;
		if (kernel_id < 10) {
			addrFile.open("../../../output/"+name+"/"+name+"_0"+std::to_string(kernel_id)+".trc");
		}
		else {
			addrFile.open("../../../output/"+name+"/"+name+"_"+std::to_string(kernel_id)+".trc");
		}
		kernel_id++;
	}
	
	// Finalise the data
	void finalise() {
		std::cout << "[Tracer] completed up to " << MAX_THREADS << " threads" << std::endl;
		finished = true;
		if( addrFile.is_open()) {
			addrFile.close();
		}
	}
	
	// Ocelot event callback
	void event(const trace::TraceEvent & event) {
		
		// Get a flat thread/block ID/dimension
		unsigned bid = event.blockId.x*event.gridDim.y*event.gridDim.z + event.blockId.y*event.gridDim.z + event.blockId.z;
		unsigned bdim = event.blockDim.x*event.blockDim.y*event.blockDim.z;
		
		// Initialise the trace
		if (!initialised) {
			addrFile << "blocksize: " << event.blockDim.x << " " << event.blockDim.y << " " << event.blockDim.z << std::endl;
			initialised = true;
		}
		
		// Finalise the trace
		if (bid == MAX_THREADS/bdim && !finished) {
			finalise();
		}
		
		// Only process the first MAX_THREADS threads
		if (bid < MAX_THREADS/bdim) {
		
			// Found a global load/store
			if (((event.instruction->addressSpace == ir::PTXInstruction::Global) &&
			    (event.instruction->opcode == ir::PTXInstruction::Ld || event.instruction->opcode == ir::PTXInstruction::St))
			   ||
			   (event.instruction->opcode == ir::PTXInstruction::Tex )) {

				// Loop over a warp's memory accesses
				for (unsigned i=0; i<event.memory_addresses.size(); i++) {
					while (event.active[i] == 0) { i++; }
					
					// Compute the address and thread ID
					unsigned long address = event.memory_addresses[i];
					unsigned gid = bid*bdim + i;
					
					// Compute the data size
					ir::PTXOperand::DataType datatype = event.instruction->type;
					unsigned vector = event.instruction->vec;
					unsigned size = vector * ir::PTXOperand::bytes(datatype);
					
					// Found a global load or texture load
					if (event.instruction->opcode == ir::PTXInstruction::Ld || event.instruction->opcode == ir::PTXInstruction::Tex) {
						loadCounter++;
						addrFile << "" << gid << " 0 " << address << " " << size << "\n";
					}
					
					// Found a global store
					if (event.instruction->opcode == ir::PTXInstruction::St) {
						storeCounter++;
						addrFile << "" << gid << " 1 " << address << " " << size << "\n";
					}
					
					// Next thread in the warp
				}
			}
			
			// Count 'compute' and 'memory' instructions to get the 'computational intensity'
			if (event.instruction->addressSpace == ir::PTXInstruction::Global) {
				ir::PTXOperand::DataType datatype = event.instruction->type;
				unsigned size = ir::PTXOperand::bytes(datatype);
				memoryCounter += size;
			}
			else {
				computeCounter++;
			}
		}
	}
};

//////////////////////////////////
// Forward declaration of the original main function
//////////////////////////////////
extern int original_main(int, char**);

//////////////////////////////////
// The new main function to call the Ocelot tracer and the original main
//////////////////////////////////
int main(int argc, char** argv) {
	TraceGenerator generator(NAME);
	ocelot::addTraceGenerator(generator);
	return original_main(argc,argv);
}

//////////////////////////////////