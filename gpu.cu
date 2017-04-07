#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256
#define CUT_OFF_SCALE 4

__constant__ const int boxesArray[9][2] = {
    {0,0},
    {-1,1},
    {0,-1},
    {1,-1},
    {1,0},
    {1,1},
    {0,1},
    {-1,1},
    {-1,0},
};

extern double size;

double sizeOfBin;
int binNumber


//  benchmarking program
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor) 
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if(r2 > cutoff*cutoff) { return; }
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //  very simple short-range repulsive force
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, int n, int* count, double sizeOfBin, int binNumber) 
{
  // Get thread (particle) ID
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   int offset = gridDim.x * blockDim.x;
   int ii = 0;
   for(ii = tid; ii < n; ii += offset) {
       particle_t p = particles[ii];
       p.ax = p.ay = 0;
       int i = int(p.x / sizeOfBin);
       int j = int(p.y / sizeOfBin);
       for(int t = 0; t < 9; t++) {
           int row = i + boxesArray[t][0];
           int col = j + boxesArray[t][1];
           if (row >= 0 && row < binNumber && col >= 0 && col < binNumber) {
               int id = row * binNumber + col;
               int start = count[id-1],end = count[id];
               for (int k = start; k < end; k++) {
                   apply_force_gpu(p, particles[k]);
               }
            }
        }
        particles[ii].ax = p.ax;
        particles[ii].ay = p.ay;
    }

}

__global__ void move_gpu (particle_t * particles, int n, double size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    int i = 0;
    for (i = tid; i < n; i += offset) {
        particle_t * p = &particles[i];
        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x  += p->vx * dt;
        p->y  += p->vy * dt;

        //
        //  bounce from walls
        //
        while( p->x < 0 || p->x > size ) {
            p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
            p->vx = -(p->vx);
        }
        while( p->y < 0 || p->y > size ) {
            p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
            p->vy = -(p->vy);
        }
    }
}

__global__ void createBins( particle_t* particles, particle_t* temp, int* count, int n, double sizeOfBin, int bin) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    int i = 0;
    for (i = threadID; i < n; i += offset) {
        int row = int(particles[i].x / sizeOfBin);
        int col = int(particles[i].y / sizeOfBin);
        int id = atomicSub(count + x * bin + y, 1);
        temp[threadID-1] = particles[i];
    }
}

__global__ void counts(particle_t* particles, int* count,int n,double sizeOfBin,int bin) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    int i = 0;
    for (i = threadID; i < n; i += offset) {
        int row = int(particles[i].x / sizeOfBin);
        int col = int(particles[i].y / sizeOfBin);
        atomicAdd(count + x * bin + y,1);
    }
}


int main( int argc, char **argv) {
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if(find_option( argc, argv, "-h" ) >= 0 ) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);

    char *savename = read_string(argc, argv, "-o", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t));

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    cudaMalloc((void **) &temp, n * sizeof(particle_t));

    set_size(n);
    sizeOfBin = cutoff * CUT_OFF_SCALE;
    binNumber = int(size / sizeOfBin) + 1;
    printf("Number of bins created ==> %d%d", binID, binID);

    int* count;
    cudaMalloc((void **) &count, (binID*binID+1) * sizeof(int));
    cudaMemset(count,0,(binNumber * binNumber + 1) * sizeof(int));
    count = count + 1;
    int* countAlloc = (int*) malloc(binNumber * binNumber * sizeof(int));

    init_particles(n, particles);

    cudaThreadSynchronize();
    double copy_time = read_timer();

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer() - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer();

    for(int step = 0; step < NSTEPS; step++) {
        //
        //  compute forces
        //
        int numOfThreads = NUM_THREADS;
        int blks = min(1024, (n + NUM_THREADS - 1) / NUM_THREADS);
        int block = blks;

        cudaMemset(count, 0, binNumber * binNumber * sizeof(int));
        counts<<<block,numOfThreads>>(d_particles, count, n, sizeOfBin, bin);

        cudaMemcpy(count, cnt, binNumber * binNumber * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 1; i < binNumber * binNumber; i++) {
            countAlloc[i] += countAlloc[i-1];
        }
        cudaMemcpy(count, countAlloc, binNumber * binNumber * sizeof(int), cudaMemcpyHostToDevice);
        createBins<<<block,numOfThreads>>>(d_particles,tmp,count,n,sizeOfBin,binNumber);
        std::swap(d_particles,temp);
        cudaMemcpy(count, countAlloc, binNumber * binNumber * sizeof(int), cudaMemcpyHostToDevice);


        compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n, count, sizeOfBin, bin);

        //
        //  move particles
        //
	    move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save(fsave, n, particles);
	    }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer() - simulation_time;

    printf("CPU-GPU copy time = %g seconds\n", copy_time);
    printf("n = %d, simulation time = %g seconds\n", n, simulation_time);

    free(particles);
    cudaFree(d_particles);
    if(fsave) { fclose(fsave); }

    return 0;
}
