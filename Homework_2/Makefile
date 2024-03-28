CXX = g++
CXXFLAGS = -Wall -g
INCLUDES = -I/usr/local/include/opencv4
LFLAGS = -L/usr/local/lib
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_ml
SRCS = main.cpp
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
