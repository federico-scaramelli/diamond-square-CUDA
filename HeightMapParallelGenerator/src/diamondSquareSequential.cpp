#include <algorithm>
#include <ctime>
#include <stdlib.h>
#include "DiamondSquareSequential.h"

void DiamondSquareSequential::InitializeDiamondSquare() {
	srand(time(nullptr));

	map[0] = rand() % 256;
	map[size - 1] = rand() % 256;
	map[size * (size - 1)] = rand() % 256;
	map[size * size - 1] = rand() % 256;
}

void DiamondSquareSequential::DiamondSquare() {

	//step = size - 1;

	while (step > 1) {
		half = step / 2;

		for (uint32_t y = 0; y < size - 1; y += step) {
			for (uint32_t x = 0; x < size - 1; x += step) {
				DiamondStep(x, y, step);
			}
		}

		//PrintMap();

		for (uint32_t x = 0; x < size; x += half) {
			for (uint32_t y = (x + half) % step; y < size; y += step) {
				SquareStep(x, y, half);
				//map[x * size + y] = 255;
			}
		}

		//PrintMap();

		randomness = std::max(randomness / 2, 1);
		step /= 2;
	}
}

void DiamondSquareSequential::DiamondStep(uint32_t x, uint32_t y, uint32_t step) {

	uint8_t value = 0;

	value = map[x * size + y] +
		map[(x + step) * size + y] +
		map[x * size + y + step] +
		map[(x + step) * size + y + step];

	value /= 4;
	value += rand() % (randomness * 2 + 1) + (-randomness);

	map[(y + half) * size + (x + half)] = value;
}

void DiamondSquareSequential::SquareStep(uint32_t x, uint32_t y, uint32_t half) {

	uint8_t value = 0;
	uint8_t count = 0;

	//std::cout << "Summing on the index [" << x << ", " << y << "]\n";
	if (y + half < size) {
		//std::cout  << "y + half: " << static_cast<int>(map[x * size + y + half]) << "\n";
		value += map[x * size + y + half];
		count++;
	}
	if (static_cast<int>(y - half) >= 0) {
		//std::cout << "y - half: " << static_cast<int>(map[x * size + y - half]) << "\n";
		value += map[x * size + y - half];
		count++;
	}
	if (x + half < size) {
		//std::cout  << "x + half: " << static_cast<int>(map[(x + half) * size + y]) << "\n";
		value += map[(x + half) * size + y];
		count++;
	}
	if (static_cast<int>(x - half) >= 0) {
		//std::cout  << "x - half: " << static_cast<int>(map[(x - half) * size + y]) << "\n";
		value += map[(x - half) * size + y];
		count++;
	}


	value /= count;
	value += rand() % (randomness * 2 + 1) + (-randomness);

	map[x * size + y] = value;
}
