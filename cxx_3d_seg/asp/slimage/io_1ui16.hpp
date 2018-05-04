#pragma once

#include <slimage/image.hpp>
#include <slimage/error.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <vector>

namespace slimage
{
	inline
	bool ReadDataLine(std::istream& is, std::string& line)
	{
		while(!is.eof()) {
			getline(is, line);
			if(line.size() > 0 && line[0] != '#') {
				return true;
			}
		}
		return false;
	}

	/** Loads a 16 bit 1-channel image from an ASCII PGM file */
	inline Image1ui16 Load1ui16(const std::string& filename) {
		if(!boost::algorithm::ends_with(filename, ".pgm")) {
			throw IoException(filename, "Load1ui16 can only handle PGM files");
		}
		std::ifstream ifs(filename);
		if(!ifs.is_open()) {
			throw IoException(filename, "Could not open file");
		}
		std::string line;
		std::vector<std::string> tokens;
		// read magic line
		ReadDataLine(ifs, line);
		boost::split(tokens, line, boost::is_any_of(" "));
		if(tokens.size() != 1 || !(tokens[0] == "P2" || tokens[0] == "P5")) {
			throw IoException(filename, "Wrong PGM file header (P2 id)");
		}
		std::string pmode = tokens[0];
		// read dimensions line
		ReadDataLine(ifs, line);
		boost::split(tokens, line, boost::is_any_of(" "));
		unsigned int w, h;
		if(tokens.size() != 2) {
			throw IoException(filename, "Wrong PGM file header (width/height)");
		}
		try {
			w = boost::lexical_cast<unsigned int>(tokens[0]);
			h = boost::lexical_cast<unsigned int>(tokens[1]);
		} catch(...) {
			throw IoException(filename, "Wrong PGM file header (width/height)");
		}
		// read max line
		ReadDataLine(ifs, line);
		boost::split(tokens, line, boost::is_any_of(" "));
		if(tokens.size() != 1 || tokens[0] != "65535") {
			throw IoException(filename, "Wrong PGM file header (max value)");
		}
		// read data
		Image1ui16 img(w, h);
		if(pmode == "P2") {
			unsigned int y = 0;
			while(ReadDataLine(ifs, line)) {
				boost::split(tokens, line, boost::is_any_of(" "));
				if(tokens.back().empty()) {
					tokens.pop_back();
				}
				if(tokens.size() != w) {
					throw IoException(filename, "Width and number of tokens in line do not match");
				}
				for(unsigned int x=0; x<w; x++) {
					img(x,y) = boost::lexical_cast<unsigned int>(tokens[x]);
				}
				y++;
			}
			if(y != h) {
				throw IoException(filename, "Height and number of lines do not match");
			}
		}
		if(pmode == "P5") {
			std::vector<char> dataline(img.size()*2);
			ifs.read(dataline.data(), img.size()*2);
			for(unsigned i=0; i<img.size(); i++) {
				uint16_t v;
				reinterpret_cast<char*>(&v)[1] = dataline[2*i];
				reinterpret_cast<char*>(&v)[0] = dataline[2*i + 1];
				img[i] = v;
			}
		}
		return img;
	}

	/** Saves a 1 channel 16 bit unsigned integer image to an ASCII PGM file */
	inline void Save(const std::string& filename, const Image1ui16& img) {
		if(!boost::algorithm::ends_with(filename, ".pgm")) {
			throw IoException(filename, "Save for 1ui16 images can only handle PGM files");
		}
		std::ofstream ofs(filename);
		ofs << "P2" << std::endl;
		ofs << img.width() << " " << img.height() << std::endl;
		ofs << "65535" << std::endl;
		for(unsigned int y=0; y<img.height(); y++) {
			for(unsigned int x=0; x<img.width(); x++) {
				ofs << img(x,y);
				if(x+1 < img.width()) {
					ofs << " ";
				}
			}
			if(y+1 < img.height()) {
				ofs << std::endl;
			}
		}
	}
}