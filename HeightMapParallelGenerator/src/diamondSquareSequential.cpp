#include "DiamondSquareSequential.h"

#pragma region Random Generator

std::random_device rd;
//std::mt19937 gen(0);
std::mt19937 generator(rd());
std::uniform_real_distribution<double> unif {-1.0, 1.0};

#pragma endregion

#pragma region Execution Functions

void DiamondSquareSequential::InitializeDiamondSquare() {
  std::cout << "Initializing Diamond Square..." << std::endl;

	/*for (int i = 0; i < 100; i++) {
		std::cout << unif(generator) << std::endl;
	}*/

	for (uint32_t x = 0; x < size; x += step) {
		for (uint32_t y = 0; y < size; y += step) {
			map[GetIndex(x, y)] = unif(generator);  
		}
	}
}

void DiamondSquareSequential::DiamondSquare() {

	while (step > 1) {
		half = step / 2;

		for (uint32_t y = half; y < size + half; y += step) {
			for (uint32_t x = half; x < size + half; x += step) {
				DiamondStep(x, y);
			}
		}

		//PrintMap();

		for (uint32_t x = 0; x < size; x += half) {
			for (uint32_t y = (x + half) % step; y < size; y += step) {
				SquareStep(x, y);
			}
		}
		
		//PrintMap();

		randomScale /= 2.0;
		step /= 2;
	}
}

void DiamondSquareSequential::DiamondStep(uint32_t x, uint32_t y) {

	double value = 0;

	value = map[GetIndex(x - half, y - half)] +
		map[GetIndex(x + half, y - half)] +
		map[GetIndex(x - half, y + half)] +
		map[GetIndex(x + half, y + half)];

	value /= 4.0;
	value += unif(generator) * randomScale;

	map[GetIndex(x, y)] = value;
}

void DiamondSquareSequential::SquareStep(uint32_t x, uint32_t y) {

	double value = map[GetIndex(x - half, y)] +
				map[GetIndex(x + half, y)] +
				map[GetIndex(x, y - half)] +
				map[GetIndex(x, y + half)];

	value /= 4.0;
	value += unif(generator) * randomScale;

	map[x * size + y] = value;
}

#pragma endregion