# compile every *.c file into *.o file
# then link them together into main.out

CC=gcc
LDFLAGS=-lopenblas
CFLAGS=-c -Wall -O3 -std=c99 -I./include
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

all: $(OBJECTS)
	$(CC) $(OBJECTS) -o main.out $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f src/*.o main.out
