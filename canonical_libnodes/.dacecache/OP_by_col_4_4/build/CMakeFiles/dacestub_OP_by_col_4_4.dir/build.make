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
CMAKE_BINARY_DIR = /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/OP_by_col_4_4/build

# Include any dependencies generated for this target.
include CMakeFiles/dacestub_OP_by_col_4_4.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dacestub_OP_by_col_4_4.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dacestub_OP_by_col_4_4.dir/flags.make

CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.o: CMakeFiles/dacestub_OP_by_col_4_4.dir/flags.make
CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.o: /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/OP_by_col_4_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.o -c /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp

CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp > CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.i

CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tdematt/src/dace/dace/codegen/tools/dacestub.cpp -o CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.s

# Object files for target dacestub_OP_by_col_4_4
dacestub_OP_by_col_4_4_OBJECTS = \
"CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.o"

# External object files for target dacestub_OP_by_col_4_4
dacestub_OP_by_col_4_4_EXTERNAL_OBJECTS =

libdacestub_OP_by_col_4_4.so: CMakeFiles/dacestub_OP_by_col_4_4.dir/tools/dacestub.cpp.o
libdacestub_OP_by_col_4_4.so: CMakeFiles/dacestub_OP_by_col_4_4.dir/build.make
libdacestub_OP_by_col_4_4.so: /usr/lib/gcc/x86_64-redhat-linux/12/libgomp.so
libdacestub_OP_by_col_4_4.so: /usr/lib64/libpthread.a
libdacestub_OP_by_col_4_4.so: CMakeFiles/dacestub_OP_by_col_4_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/OP_by_col_4_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libdacestub_OP_by_col_4_4.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dacestub_OP_by_col_4_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dacestub_OP_by_col_4_4.dir/build: libdacestub_OP_by_col_4_4.so

.PHONY : CMakeFiles/dacestub_OP_by_col_4_4.dir/build

CMakeFiles/dacestub_OP_by_col_4_4.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dacestub_OP_by_col_4_4.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dacestub_OP_by_col_4_4.dir/clean

CMakeFiles/dacestub_OP_by_col_4_4.dir/depend:
	cd /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/OP_by_col_4_4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tdematt/src/dace/dace/codegen /home/tdematt/src/dace/dace/codegen /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/OP_by_col_4_4/build /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/OP_by_col_4_4/build /home/tdematt/src/dse_with_dace/canonical_libnodes/.dacecache/OP_by_col_4_4/build/CMakeFiles/dacestub_OP_by_col_4_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dacestub_OP_by_col_4_4.dir/depend

