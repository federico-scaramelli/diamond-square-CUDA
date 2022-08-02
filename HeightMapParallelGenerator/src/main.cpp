#include "bmpHandler.h"
#include "DiamondSquareSequential.h"

int main(int argc, char** argv) {
  	const uint32_t size = 4097;

	try {
		DiamondSquareSequential ds{size};
		ds.SetRandomScale(15.0);
		ds.SetInitialStepSize(2048);
		ds.ExecuteDiamondSquare();
		//ds.PrintMap();
		//ds.PrintGrayScaleMap();
		ds.SaveGrayScaleImage("map.bmp", 1);
		ds.SaveColorImage("mapColor.bmp", 1);
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return(0);
	}

	return(0);
}