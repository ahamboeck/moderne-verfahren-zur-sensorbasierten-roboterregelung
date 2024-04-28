CXX = g++
CXXFLAGS = -Wall -g
INCLUDES = -I/usr/local/include/opencv4 -I/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/siftDetectorTrain
LFLAGS = -L/usr/local/lib
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_features2d -lopencv_xfeatures2d
SRCS = main.cpp siftDetectorTrain/siftDetectorTrain.cpp
OBJS = $(SRCS:.cpp=.o)
MAIN = main

.PHONY: clean

all: $(MAIN)
	@echo Simple compiler named main has been compiled

$(MAIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)
