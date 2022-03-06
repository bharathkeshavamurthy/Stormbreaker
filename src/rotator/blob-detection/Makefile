CC = "C:/Program Files (x86)/CodeBlocks/MinGW/bin/g++.exe"
RM = del
SRC = main.cpp CommunicationMessage.cpp CommunicationNMEAInterface.cpp NetworkConnection.cpp Utility.cpp MySwisTrackClient.cpp PositionController.cpp
TARGET = main
INCLUDES =
LIBS =
CFLAGS = $(INCLUDES) $(LIBS) -std=c++11
LFLAGS = -lpthread -lws2_32
DEPS = CommunicationMessage.h CommunicationNMEAInterface.h NetworkConnection.h Utility.h MySwisTrackClient.h PositionController.h
OBJS =

%.o: %.c $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS)

all: $(OBJS)
	$(CC) $(SRC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LFLAGS)

clean:
	$(RM) -f $(TARGET)
	$(RM) *.o