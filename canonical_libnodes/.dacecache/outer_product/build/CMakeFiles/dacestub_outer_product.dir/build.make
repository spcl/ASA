# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/tdematt/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/tdematt/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tdematt/src/dace/dace/codegen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/outer_product/build

# Include any dependencies generated for this target.
include CMakeFiles/dacestub_outer_product.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dacestub_outer_product.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dacestub_outer_product.dir/flags.make

CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.o: CMakeFiles/dacestub_outer_product.dir/flags.make
CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.o: /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/outer_product/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.o -c /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp

CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp > CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.i

CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp -o CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.s

# Object files for target dacestub_outer_product
dacestub_outer_product_OBJECTS = \
"CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.o"

# External object files for target dacestub_outer_product
dacestub_outer_product_EXTERNAL_OBJECTS =

libdacestub_outer_product.so: CMakeFiles/dacestub_outer_product.dir/tools/dacestub.cpp.o
libdacestub_outer_product.so: CMakeFiles/dacestub_outer_product.dir/build.make
libdacestub_outer_product.so: /usr/lib/gcc/x86_64-redhat-linux/12/libgomp.so
libdacestub_outer_product.so: /usr/lib64/libpthread.a
libdacestub_outer_product.so: CMakeFiles/dacestub_outer_product.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/outer_product/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libdacestub_outer_product.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dacestub_outer_product.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dacestub_outer_product.dir/build: libdacestub_outer_product.so

.PHONY : CMakeFiles/dacestub_outer_product.dir/build

CMakeFiles/dacestub_outer_product.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dacestub_outer_product.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dacestub_outer_product.dir/clean

CMakeFiles/dacestub_outer_product.dir/depend:
	cd /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/outer_product/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tdematt/src/dace/dace/codegen /home/tdematt/src/dace/dace/codegen /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/outer_product/build /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/outer_product/build /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/outer_product/build/CMakeFiles/dacestub_outer_product.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dacestub_outer_product.dir/depend

