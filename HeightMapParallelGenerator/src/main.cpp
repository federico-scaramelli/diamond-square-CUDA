
#include <iostream>
#include "bmpHandler.h"

int main(int argc, char** argv) {

	Color color(std::string("0xFF0000"));
	Color color2("0x0000FF");
	Color colorW(true);
	Color colorB(false);

	BMP image(800, 600);
	image.fillRegion(50, 20, 100, 200, color, 255);
	image.fillRegion(200, 200, 100, 200, color2, 255);
	image.fillRegion(300, 400, 100, 20, colorW, 255);
	image.fillRegion(500, 500, 20, 20, colorB, 255);

	image.write("prova.bmp");

	BMP image2("prova.bmp");
	image2.fillRegion(600, 100, 200, 300, 255, 0, 0, 255);
	image2.write("prova.bmp");

	return(0);
}
