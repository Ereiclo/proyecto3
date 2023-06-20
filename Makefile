
library_development:
	g++ library_development.cpp -o exe -std=c++11 -larmadillo -g

run: library_development
	./exe