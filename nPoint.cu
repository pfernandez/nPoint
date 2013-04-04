/*	Simulation overview:

	1. Define the information contained within a point in space as a c++ struct.
	2. Generate multiple copies of this point in a GPU array, each with unique parameters.
	3. Calculate a quantity at each point due to the given parameters within a CUDA kernel.
	4. Advance in time, and repeat.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda.h>



int numPoints = 4096,		// number of points to generate (multiples of 32 are best)
    maxSteps  = 250000, 	// total iterations
	saveSteps = 1000;		// data will be written to file when current step % saveSteps = 0


__device__ __constant__ float dt = 0.0005,   // Time step size.
                              A, k, w, c, dk;

const float pi = 3.14159265358979f;                          
const float plotBoundary = 4 * pi;


typedef struct
{
	// Represents a point in space with its associated fields, as well
	// as any information we wish to carry along with each point.
	
	float x, y,     // position data
		  psi;      // wave height
		  
} point;



// HOST FUNCTIONS //////////////////////////////////////////////////////////////////////////////////
//


// Get GPU debugging mesagges.
inline void check_cuda_errors( const char *filename, const int line_number )
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if( error != cudaSuccess )
  {
    printf( "CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString( error ) );
    exit( -1 );
  }
#endif
}

// Get the current timestamp.
char *date_and_time()
{
	time_t  time_raw_format;
	struct tm * ptr_time;
	time ( &time_raw_format);
	ptr_time = localtime ( &time_raw_format );
	return asctime(ptr_time);
}

// Store simulation data into the directory specified by *path.
void writeDataToFile( char *path, point *p, int numPoints, float simTime, float plotLimit )
{
	// Initialize data file and write common parameters:
	FILE *file;	
	char filename[ strlen( path )+24 ];
	sprintf( filename,"%s/%.16f.dat", path, simTime );
	file = fopen( filename,"w" );
	char sp[10] = "         ";
	fprintf( file,"# limit = %g\n#\n# x%s  y%s  psi\n", plotLimit, sp, sp);
	
	// Write data to file:
	for( int i = 0; i < numPoints; i++ )
	{
		fprintf( file,"% .4e % .4e % .4e\n",
			p[i].x, p[i].y, p[i].psi );
	}
	
	fclose(file);
}


// Calculate constants used in device kernels.
void initializeDeviceConstants()
{
    float a = 1,
          lambda = 2 * pi,
          K = 2 * pi / lambda,
          hBar = 1 / ( 2 * pi ),  // letting h = 1
          m = 1,
          W = 0.5 * hBar * K * K / m,
          C =  W / K,
          DK = K / 8;  // determines the "tightness" of the wave packet

    cudaMemcpyToSymbol( A, &a, sizeof( float ) );
    cudaMemcpyToSymbol( k, &K, sizeof( float ) );
    cudaMemcpyToSymbol( w, &W, sizeof( float ) );
    cudaMemcpyToSymbol( c, &C, sizeof( float ) );
    cudaMemcpyToSymbol( dk, &DK, sizeof( float ) );
}


// Populate the point array on the host.
void initializePoints( point *p, int N )
{
    // Define the boudaries of the point-space.
    float dR = plotBoundary / sqrt( (float)N ),
          dTheta = 2 * pi / sqrt( (float)N );
          
    // Assign each point a specific location in space.
    int i = 0;
	for( float r = 0; r <= plotBoundary; r += dR )
	{	
	    for( float theta = 0; theta <= 2 * pi; theta += dTheta )
	    {
	        if( i > N )
	            continue;
            p[i].x = r * cos( theta );     // position vector
            p[i].y = r * sin( theta );	   //
            i++;
        }
	}
}

//
// END HOST FUNTIONS ///////////////////////////////////////////////////////////////////////////////




// DEVICE KERNELS //////////////////////////////////////////////////////////////////////////////////
//


__device__ int getGlobalIndex( void ) { return blockIdx.x * blockDim.x + threadIdx.x; }

__device__ float t = 0;
                              
__device__ void updateTime() { t += dt; }

// Calcutate the new point values in time and space at each timestep.
__global__ void incrementPoint( point *p, int numPoints )
{
	int	i = getGlobalIndex();
	 
    float r = sqrt( p[i].x*p[i].x + p[i].y*p[i].y );
    
    //p[i].psi = A * cos( k * r - w * t );  // simple circular wave
    
    // Disbursive Wave Packet
    // From A. C. Philips: Intro to Quantum Mechanics, problem 2.2 and fig. 2.1, with c = w/k.
    float alpha = r - c * t;
    p[i].psi = 2 * A * dk * sin( dk * alpha ) * cos( k * alpha ) / ( dk * alpha );
    
    updateTime();
}


//
// END DEVICE KERNELS //////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[])
{


////////////////////////////////////////////////////////////////////////////////////////////////////
// SELECT DEVICE


    // Get the number of CUDA enabled devices.
	int deviceCount;
	cudaGetDeviceCount( &deviceCount );
	if( deviceCount < 1 )
	{
		printf( "No CUDA-capable devices were detected.\n" );
		return 1;
	}

	// Get properties, select device, and display info.
	int device, useDevice;
	float computeCapability, cudaVersion = 0;
	cudaDeviceProp deviceProp;
	for( device = 0; device < deviceCount; device++ )
	{
	    // Display info for each device.
		cudaGetDeviceProperties( &deviceProp, device );
		computeCapability = deviceProp.major + deviceProp.minor * 0.1;
			
	    // Select the device with the greatest compute capability. If multiple devices
	    // with this capability are found, the first encountered will be used.
		if( computeCapability > cudaVersion )
		{
			cudaVersion = computeCapability;
			useDevice = device;
		}
	}
	device = useDevice;
	cudaGetDevice(&device);
	cudaGetDeviceProperties( &deviceProp, device );
    printf( "# Using %s, device number: %d, compute capability: %g.\n",
        deviceProp.name, device, cudaVersion );

    // Check for kernel timeout.
	if( deviceProp.kernelExecTimeoutEnabled == 1 )
	{
	    printf( "#\n" );
		printf( "# Warning: Timeout enabled. Using this device without a display \n" );
		printf( "# may lift kernel timeout restrictions.\n" );
	}  
	
	
// END SELECT DEVICE
////////////////////////////////////////////////////////////////////////////////////////////////////


	// Allocate memory:
	int threadsPerBlock = ( deviceProp.major >= 2 ? 512 : 256 ),
		numBlocks		= ceil( (float)numPoints / (float)threadsPerBlock ),
		numBytes		= numPoints * sizeof( point );
	point *cpuPointArray, *gpuPointArray;
	cpuPointArray = (point*)malloc( numBytes );
	cudaMalloc( (void**)&gpuPointArray, numBytes );
	check_cuda_errors( __FILE__, __LINE__ );
	
	
////////////////////////////////////////////////////////////////////////////////////////////////////
// BEGIN SIMULATION


	// Begin simulation:
	for( int s = 0; s < maxSteps; s++ )
	{
		
		// Perform setup tasks before the first iteration.
		if( s == 0 )
		{	
		    // Initialize point array and device constants.
			srand( time( NULL ) );
			initializePoints( cpuPointArray, numPoints );
			initializeDeviceConstants();
			cudaMemcpy( gpuPointArray, cpuPointArray, numBytes, cudaMemcpyHostToDevice );

			// Write simulation info to stdout.
			float stepSize;
			cudaMemcpyFromSymbol( &stepSize, dt, sizeof(float) );
			printf( "#\n#\n" );
			printf( "# Started %s#\n# numPoints = %d\n# maxSteps  = %d\n# dt        = %g\n#\n",
				date_and_time(), numPoints, maxSteps, stepSize );
			printf( "# Steps   | Seconds  | Sim. Time\n# ------------------------------\n" );
			
			check_cuda_errors(__FILE__, __LINE__);	
		}

		// Execute portion of simulation on GPU:
		incrementPoint<<<numBlocks,threadsPerBlock>>>( gpuPointArray, numPoints );
			
		// Retrieve results:
		float simTime;
		cudaMemcpyFromSymbol( &simTime, t, sizeof( float ) );
		cudaMemcpy( cpuPointArray, gpuPointArray, numBytes, cudaMemcpyDeviceToHost );

		// Output data every [saveSteps] steps.
		if( s % saveSteps == 0 )
		{
		    // Write to file if save paths specified in argument to main().
		    if( argc > 1 )
			    writeDataToFile( argv[1], cpuPointArray, numPoints, simTime, plotBoundary );
			
			// Print data to stdout.
			printf( "%.4e %.4e %.4e\n", (float)s, (float)( clock() / CLOCKS_PER_SEC ), simTime );
		}

	}
	
	
// END SIMULATION
////////////////////////////////////////////////////////////////////////////////////////////////////
 	
 	
  	// Deallocate memory and exit.
  	free( cpuPointArray );
  	cudaFree( gpuPointArray );
  	check_cuda_errors( __FILE__, __LINE__ );
  	printf( "\n# Completed %s", date_and_time() );
	return 0;
	
	
}	// end main
