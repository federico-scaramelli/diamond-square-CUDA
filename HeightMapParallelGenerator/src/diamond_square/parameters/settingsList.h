#pragma once
#include <cstdint>

struct DiamondSquareSetting {
public:
	uint32_t size;
	uint32_t initialStepSize;
	float randomScale;
	uint8_t imageTileSize;
};

enum DiamondSquareSettings {
	Size9_Step8_Rnd1,
	Size65_Step64_Rnd1,
	Size129_Step128_Rnd2,
	Size257_Step256_Rnd3,
	Size513_Step256_Rnd5,
	Size1025_Step512_Rnd10,
	Size2049_Step1024_Rnd10,
	Size4097_Step1024_Rnd15,
	Size4097_Step2048_Rnd15,
	Size8193_Step2048_Rnd15,
	Size8193_Step2048_Rnd20,
	Size16385_Step4096_Rnd30
};

DiamondSquareSetting diamondSquareSettings[] = {
  {9, 8, 1.0, 1},
  {65, 64, 1.0, 1},
  {129, 128, 2.0, 1},
  {257, 256, 3.0, 1},
  {513, 256, 5.0, 1},
  {1025, 512, 10.0, 1},
  {2049, 1024, 10.0, 1},
  {4097, 1024, 15.0, 1},
  {4097, 2048, 15.0, 1},
  {8193, 2048, 15.0, 1},
  {8193, 2048, 20.0, 1},
  {16385, 4096, 30.0, 1}
};