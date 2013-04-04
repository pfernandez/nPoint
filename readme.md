#nPoint

A simple, equation-agnostic physics simulator for the Nvidia GPU. Requires the nvcc compiler.

##Simulation overview:

1. Define the information contained within a point in space as a c++ struct.
2. Generate multiple copies of this point in a GPU array, each with unique parameters.
3. Calculate a quantity at each point due to the given parameters within a CUDA kernel.
4. Advance in time, and repeat.

See runit.sh for more info on how to run the application as well as options for
auto-plotting (requires gnuplot) and movie generation (requires vlc).

nPoint.cu is well commented, and includes a simple wave-packet equation as an
example simulation.
