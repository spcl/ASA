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
CMAKE_SOURCE_DIR = /home/tdematt/work/dace/dace/codegen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tdematt/work/canonical_libnodes/.dacecache/pure_2_2_2/build

# Include any dependencies generated for this target.
include CMakeFiles/dacestub_pure_2_2_2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dacestub_pure_2_2_2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dacestub_pure_2_2_2.dir/flags.make

CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.o: CMakeFiles/dacestub_pure_2_2_2.dir/flags.make
CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.o: /home/tdematt/work/dace/dace/codegen/tools/dacestub.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tdematt/work/canonical_libnodes/.dacecache/pure_2_2_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.o -c /home/tdematt/work/dace/dace/codegen/tools/dacestub.cpp

CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tdematt/work/dace/dace/codegen/tools/dacestub.cpp > CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.i

CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tdematt/work/dace/dace/codegen/tools/dacestub.cpp -o CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.s

# Object files for target dacestub_pure_2_2_2
dacestub_pure_2_2_2_OBJECTS = \
"CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.o"

# External object files for target dacestub_pure_2_2_2
dacestub_pure_2_2_2_EXTERNAL_OBJECTS =

libdacestub_pure_2_2_2.so: CMakeFiles/dacestub_pure_2_2_2.dir/tools/dacestub.cpp.o
libdacestub_pure_2_2_2.so: CMakeFiles/dacestub_pure_2_2_2.dir/build.make
libdacestub_pure_2_2_2.so: /usr/lib/gcc/x86_64-redhat-linux/11/libgomp.so
libdacestub_pure_2_2_2.so: /usr/lib64/libpthread.a
libdacestub_pure_2_2_2.so: CMakeFiles/dacestub_pure_2_2_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tdematt/work/canonical_libnodes/.dacecache/pure_2_2_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libdacestub_pure_2_2_2.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dacestub_pure_2_2_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dacestub_pure_2_2_2.dir/build: libdacestub_pure_2_2_2.so

.PHONY : CMakeFiles/dacestub_pure_2_2_2.dir/build

CMakeFiles/dacestub_pure_2_2_2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dacestub_pure_2_2_2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dacestub_pure_2_2_2.dir/clean

CMakeFiles/dacestub_pure_2_2_2.dir/depend:
	cd /home/tdematt/work/canonical_libnodes/.dacecache/pure_2_2_2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tdematt/work/dace/dace/codegen /home/tdematt/work/dace/dace/codegen /home/tdematt/work/canonical_libnodes/.dacecache/pure_2_2_2/build /home/tdematt/work/canonical_libnodes/.dacecache/pure_2_2_2/build /home/tdematt/work/canonical_libnodes/.dacecache/pure_2_2_2/build/CMakeFiles/dacestub_pure_2_2_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dacestub_pure_2_2_2.dir/depend

