##################################
##
## == A reuse distance based GPU cache model
## This file is part of a cache model for GPUs. The cache model is based on
## reuse distance theory extended to work with GPUs. The cache model primarly
## focusses on modelling NVIDIA's Fermi architecture.
##
## == File details
## Filename...........Makefile
## Author.............Cedric Nugteren <www.cedricnugteren.nl>
## Affiliation........Eindhoven University of Technology, The Netherlands
## Last modified on...30-Oct-2013
##
##################################

##################################
## Settings and constants
##################################

# Set the compiler and compiler flags
RM             = rm -f
CXX            = g++
CXXNEW         = /usr/bin/g++-4.7
CXXFLAGS       = -O3 -m64 -std=c++0x -Wall
CUDAINCLUDE    = -I/usr/local/cuda/include/
NVCC           = nvcc
NVCCFLAGS      = -O3 -m64 -arch=sm_20
R              = Rscript

# Set the NVIDIA profiler to output cache miss rates
NVPROF         = nvprof --devices 0 --events l1_global_load_hit,l1_global_load_miss --aggregate-mode-off

# Set the directories
MODEL_DIR      = src/model
TRACER_DIR     = src/tracer
VISUALISER_DIR = src/visualiser
PROFILER_DIR   = src/profiler
TEMP_DIR       = temp
BIN_DIR        = bin
OUTPUT_DIR     = output

# Set the stack size to unlimited
ULIMIT         = ulimit -s unlimited

##################################
## Remote execution
##################################

# Set a GPU server for remote execution: this can be localhost if a GPU is available
GPU_SERVER     = localhost

##################################
## Cache model targets
##################################

# Build and run the model
all: run

# Build the cache model
build: $(MODEL_DIR)/*.cpp $(MODEL_DIR)/*.h
	@echo "= Building the cache model ="
	$(CXXNEW) $(CXXFLAGS) $(MODEL_DIR)/*.cpp -o $(BIN_DIR)/cachemodel

# Build and run (NAME as argument)
run: name build
	@echo "= Running the cache model ="
	@mkdir -p $(OUTPUT_DIR)/${NAME}
	$(BIN_DIR)/cachemodel ${NAME}

##################################
## Tracer targets
##################################

# Generate a trace (NAME and DIR as arguments)
# Note: this assumes GPU-Ocelot is installed
trace: tracer
	@echo "= Generating a trace ="
	@mkdir -p $(OUTPUT_DIR)/${NAME}
	cd $(TEMP_DIR)/tracer/${NAME}/ && $(ULIMIT) && ./${NAME}

# Build a CUDA benchmark with the tracer included (NAME and DIR as arguments)
# Note: this assumes GPU-Ocelot is installed
tracer: name dir
	@mkdir -p $(TEMP_DIR)/tracer/${NAME}
	@cp ${DIR}/* $(TEMP_DIR)/tracer/${NAME}
	@cp $(TRACER_DIR)/tracer.cpp $(TEMP_DIR)/tracer/${NAME}
	@cp $(TRACER_DIR)/configure.ocelot $(TEMP_DIR)/tracer/${NAME}
	cd $(TEMP_DIR)/tracer/${NAME}/ && $(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) --cuda -o $(NAME).cu.cpp $(NAME).cu -Dmain=original_main -D$(INPUT_SIZE) -D$(CACHE_SIZE) -DEMULATION
	cd $(TEMP_DIR)/tracer/${NAME}/ && $(CXX) $(CXXFLAGS) $(CUDAINCLUDE) -c -Wno-unused-but-set-variable -o ${NAME}.cu.o ${NAME}.cu.cpp
	cd $(TEMP_DIR)/tracer/${NAME}/ && $(CXX) $(CXXFLAGS) -c -o tracer.o tracer.cpp -DNAME='"${NAME}"'
	cd $(TEMP_DIR)/tracer/${NAME}/ && $(CXX) -o ${NAME} ${NAME}.cu.o tracer.o `OcelotConfig -l -t`

##################################
## Verification targets
##################################

# Verify the cache miss rate on the hardware (NAME and DIR as arguments)
# Note: this assumes a GPU is available in the system and CUDA/NVCC is installed
verify: verifier
	@echo "= Verifying the cache miss rate on the hardware ="
	@mkdir -p $(TEMP_DIR)/verifier/
	@mkdir -p $(OUTPUT_DIR)/${NAME}
	ssh $(GPU_SERVER) 'cd $(TEMP_DIR)/verifier/${NAME}/ && $(ULIMIT) && $(NVPROF) ./${NAME}' > $(TEMP_DIR)/verifier/${NAME}.prof
	ruby $(PROFILER_DIR)/parse_profile.rb $(TEMP_DIR)/verifier/${NAME}.prof $(OUTPUT_DIR)/${NAME}/ ${NAME}

# Build a CUDA benchmark to execute on the hardware (NAME and DIR as arguments)
# Note: this assumes a GPU is available in the system and CUDA/NVCC is installed
verifier: name dir
	@echo "= Building '${NAME}' for execution on the hardware ="
	@ssh $(GPU_SERVER) 'mkdir -p $(TEMP_DIR)/verifier/${NAME}/'
	@scp -q ${DIR}/* $(GPU_SERVER):$(TEMP_DIR)/verifier/${NAME}/
	ssh $(GPU_SERVER) 'cd $(TEMP_DIR)/verifier/${NAME}/ && $(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -o ${NAME} ${NAME}.cu -D$(INPUT_SIZE) -D$(CACHE_SIZE)'

##################################
## Visualisation targets
##################################

# Run the Rscript to generate a histogram (NAME as argument)
# Note: this assumes R is installed
histogram: name
	@echo "= Generating a histogram ="
	$(R) $(VISUALISER_DIR)/histogram.r ${NAME}

##################################
## Change between cache configurations
##################################

# Change the cache configuration
go16:
	cp configurations/default16.conf configurations/current.conf

# Change the cache configuration
go48:
	cp configurations/default48.conf configurations/current.conf

##################################
## Targets to check the proper inclusion of arguments
##################################

# Check for the "NAME" argument
name:
ifeq (${NAME},)
	@echo "Please provide NAME='...' to the command line as argument"
	false
endif

# Check for the "DIR" argument
dir:
ifeq (${DIR},)
	@echo "Please provide DIR='...' to the command line as argument"
	false
endif

##################################
## Maintenance targets
##################################

# Clean everything
clean:
	@echo "= Cleaning ="
	$(RM) $(BIN_DIR)/cachemodel
	$(RM) -r $(TEMP_DIR)

# Make it really clean (also delete the produced output)
realclean: clean
	$(RM) -r $(OUTPUT_DIR)

##################################