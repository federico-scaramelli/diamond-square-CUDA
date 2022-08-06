#include "diamondSquareParallel.h"

#pragma region CheckCUDACalls

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#pragma endregion

#pragma region Random Generator

#include "curand.h"

void DiamondSquareParallel::GenerateRandomNumbers()
{
    int seed = random_int_uniform();
	curandGenerator_t generator;
	CHECK_CURAND(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937))
	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator, seed))

    // Allocate memory on the host
    randoms = new float[totalSize];

    /* Allocate n floats on device */
    CHECK(cudaMalloc((void **)&devData, totalSize * sizeof(float)))

    /* Generate n floats on device */
    CHECK_CURAND(curandGenerateUniform(generator, devData, totalSize))

    /* Copy device memory to host */
    CHECK(cudaMemcpy(randoms, devData, totalSize * sizeof(float), cudaMemcpyDeviceToHost))

    auto count = 0;
    /* Show result */
    for(int i = 0; i < totalSize; i++) {
        count = getRandom(randoms + i) ? count + 1 : count;
        //std::cout << randoms[i];
    }
    std::cout << count << " negativi" << std::endl;
    std::cout << totalSize - count << " positivi" << std::endl;
    /* Cleanup */
    CHECK_CURAND(curandDestroyGenerator(generator))
    delete[] randoms;  
}

inline bool DiamondSquareParallel::getRandom(float* const value)
{
    bool cond = static_cast<int>(*value * 10) / 1 & 0x01;
    *value = *value * (-1) * cond + *value * !cond;
    return cond;
}


#pragma endregion

void DiamondSquareParallel::InitializeDiamondSquare()
{
    GenerateRandomNumbers();

	//If less than epsilon do it on the CPU
	//otherwise invoke the kernel InitializeDiamondSquareParallel

}


void DiamondSquareParallel::DiamondSquare() {
	while (step > 1) {
		half = step / 2;

		//DiamondStep();

		//PrintMap();

		//SquareStep();

		//PrintMap();

		randomScale /= 2.0f;
		step /= 2;
	}

    CleanUp();
}

void DiamondSquareParallel::DiamondStep() {
	
}

void DiamondSquareParallel::SquareStep() {
	
}

void DiamondSquareParallel::CleanUp() {
    CHECK(cudaFree(devData))
}


__global__ void InitializeDiamondSquareParallel() {
	
}


__global__ void DiamondStepParallel() {
	
}

__global__ void SquareStepParallel() {
	
}