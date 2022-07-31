#pragma once
#include <cstdint>
#include <iostream>
#include <sstream>
#include "utils.h"

struct ColorPixel {
public:
	uint8_t B;
	uint8_t G;
	uint8_t R;

	ColorPixel(const char* hexString) {
		if(hexString[0] == '#') {
			hexString++;
		}

		int n;
		std::istringstream(hexString) >> std::hex >> n;

		R = n >> 16 & 0xff;
		//std::cout << (int)R << std::endl;
		G = n >> 8 & 0xff;
		//std::cout << (int)G << std::endl;
		B = n & 0xff;
		//std::cout << (int)B << std::endl;
	}

	ColorPixel(const bool white) {
		if (white) {
			this->B = 255;
			this->G = 255;
			this->R = 255;	
		} else {
			this->B = 0;
			this->G = 0;
			this->R = 0;	
		}
	}

	ColorPixel(uint32_t B, uint32_t G, uint32_t R) {
		this->B = B;
		this->G = G;
		this->R = R;
	}

	ColorPixel(double B, double G, double R, double min, double max) {
		/*this->B = static_cast<uint8_t> ((B + 1.0) * 255.0 / 2.0);
		this->G = static_cast<uint8_t> ((G + 1.0) * 255.0 / 2.0);
		this->R = static_cast<uint8_t> ((R + 1.0) * 255.0 / 2.0);*/

		this->B = mapValue(min, max, 0, 255, B);
		this->G = mapValue(min, max, 0, 255, G);
		this->R = mapValue(min, max, 0, 255, R);
	}

	ColorPixel() {
		this->B = 0;
		this->G = 0;
		this->R = 0;	
	}

	friend std::ostream& operator<<(std::ostream& os, const ColorPixel& c)
	{
	    os << "(" << static_cast<int>(c.B) << ")";
	    return os;
	}
};