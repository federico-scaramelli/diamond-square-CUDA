#pragma once
#include <cstdint>

struct DiamondSquareSetting {
public:
	uint32_t size;
	uint32_t initialStepSize;
	double randomScale;
	uint8_t imageTileSize;
};

DiamondSquareSetting diamondSquareSettings[] = {
  {9, 8, 1.0, 1},
  {513, 256, 5.0, 1},
  {1025, 512, 10.0, 1},
  {2049, 1024, 10.0, 1},
  {4097, 1024, 15.0, 1},
  {4097, 2048, 15.0, 1},
  {8193, 2048, 15.0, 1},
  {8193, 2048, 20.0, 1},
  {16385, 4096, 30.0, 1}
};