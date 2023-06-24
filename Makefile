
library_development:
	g++ library_development.cpp -o exe -std=c++11 -larmadillo -g


classification:
	g++ classification_model.cpp -o exe  -std=c++11 -larmadillo -g

run: classification
	./exe