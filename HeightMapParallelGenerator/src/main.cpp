#include "bmpHandler.h"
#include "DiamondSquareSequential.h"

int main(int argc, char** argv) {

	/*
	 * TODO: Change grayscale generation -> the array should be of int value instead of Color value
	 * TODO: Set color extention
	 * TODO: Change stepSize setting with dedicated setter
	 */

	const uint32_t size = 16385;

	try {
		DiamondSquareSequential ds{size};
		ds.SetRandomScale(30.0);
		ds.ExecuteDiamondSquare(4096);
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