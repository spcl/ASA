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
CMAKE_BINARY_DIR = /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/build

# Include any dependencies generated for this target.
include CMakeFiles/seq_1_5.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/seq_1_5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/seq_1_5.dir/flags.make

CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.o: CMakeFiles/seq_1_5.dir/flags.make
CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.o: /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.o -c /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp

CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp > CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.i

CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp -o CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.s

# Object files for target seq_1_5
seq_1_5_OBJECTS = \
"CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.o"

# External object files for target seq_1_5
seq_1_5_EXTERNAL_OBJECTS =

libseq_1_5.so: CMakeFiles/seq_1_5.dir/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/src/cpu/seq_1_5.cpp.o
libseq_1_5.so: CMakeFiles/seq_1_5.dir/build.make
libseq_1_5.so: /usr/lib/gcc/x86_64-redhat-linux/12/libgomp.so
libseq_1_5.so: /usr/lib64/libpthread.a
libseq_1_5.so: CMakeFiles/seq_1_5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libseq_1_5.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/seq_1_5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/seq_1_5.dir/build: libseq_1_5.so

.PHONY : CMakeFiles/seq_1_5.dir/build

CMakeFiles/seq_1_5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/seq_1_5.dir/cmake_clean.cmake
.PHONY : CMakeFiles/seq_1_5.dir/clean

CMakeFiles/seq_1_5.dir/depend:
	cd /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tdematt/src/dace/dace/codegen /home/tdematt/src/dace/dace/codegen /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/build /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/build /home/tdematt/src/dse_with_dace/tests/nodes/.dacecache/seq_1_5/build/CMakeFiles/seq_1_5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/seq_1_5.dir/depend

