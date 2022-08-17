#pragma once
#include "../diamondSquareBase.h"
#include "../../utils/utils.h"


class DiamondSquareSequential : public DiamondSquareBase {
public:
	//Constructor
	explicit DiamondSquareSequential(uint32_t size) : DiamondSquareBase(size) {}

#pragma region Execution Functions

	void InitializeDiamondSquare() override;

	void DiamondSquare() override;

	void DiamondStep() override;
	
	void SquareStep() override;


#pragma endregion
};