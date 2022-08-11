#pragma once
#include "diamondSquareParallel.h"

class DiamondSquareParallel_Constant : public DiamondSquareParallel
{
public:
	DiamondSquareParallel_Constant(uint32_t size) : DiamondSquareParallel(size) {}

	void InitializeDiamondSquare() override;
	void DiamondSquare() override;
	void DiamondStep() override;
	void SquareStep() override;
};