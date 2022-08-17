#include "diamondSquareSequential.h"
#include "../parameters/applicationSettings.h"

#pragma region Execution Functions

void DiamondSquareSequential::InitializeDiamondSquare ()
{
	std::cout << " ==== SEQUENTIAL DIAMOND SQUARE ==== "<< std::endl << std::endl;
	std::cout << " - INITIALIZATION - " << std::endl;
	std::cout << "Initializing Diamond Square [" << size << " x " << size << "]..." << std::endl;

	for (uint32_t x = 0; x < size; x += step) {
		for (uint32_t y = 0; y < size; y += step) {
			map[GetIndexOnHost (x, y)] = RandomFloatUniform();
		}
	}
}

void DiamondSquareSequential::DiamondSquare ()
{
	while (step > 1) {
		half = step / 2;

		DiamondStep();

#if PRINT_DIAMOND_STEP_SEQ
		PrintFloatMap();
#endif

		SquareStep();

#if PRINT_SQUARE_STEP_SEQ
		PrintFloatMap();
#endif

		randomScale /= 2.f;
		step /= 2;
	}
}

void DiamondSquareSequential::DiamondStep ()
{
	for (uint32_t x = half; x < size; x += step) {
		for (uint32_t y = half; y < size; y += step) {
			float value = 0;

			value = map[GetIndexOnHost (x - half, y - half)] +
				map[GetIndexOnHost (x + half, y - half)] +
				map[GetIndexOnHost (x - half, y + half)] +
				map[GetIndexOnHost (x + half, y + half)];

			value /= 4.0f;
			value += RandomFloatUniform() * randomScale;

			map[GetIndexOnHost (x, y)] = value;
		}
	}
}

void DiamondSquareSequential::SquareStep ()
{
	for (int x = 0; x < size; x += half) {
		for (int y = (x + half) % step; y < size; y += step) {

			uint32_t idx = 0;
			float value = 0;
			int count = 0;

			idx = (x - half) * size + y;
			if (idx < totalSize && idx >= 0) {
				value += map[idx];
				count++;
			}
			idx = (x + half) * size + y;
			if (idx < totalSize && idx >= 0) {
				value += map[idx];
				count++;
			}
			idx = x * size + y - half;
			if (idx < totalSize && idx >= 0 && static_cast<int> (y - half) >= 0) {
				value += map[idx];
				count++;
			}
			idx = x * size + y + half;
			if (idx < totalSize && idx >= 0 && y + half < size) {
				value += map[idx];
				count++;
			}

			/*float value = map[GetIndexOnHost(x - half, y)] +
				map[GetIndexOnHost(x + half, y)] +
				map[GetIndexOnHost(x, y - half)] +
				map[GetIndexOnHost(x, y + half)];*/
			
			value /= count;
			value += RandomFloatUniform() * randomScale;

			map[GetIndexOnHost (x, y)] = value;
		}
	}
}

#pragma endregion
