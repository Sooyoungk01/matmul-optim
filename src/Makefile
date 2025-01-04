# compile every *.c file into *.o file
# then link them together into main.out

CC=gcc
LDFLAGS=-lopenblas
CFLAGS=-c -Wall -O3 -std=c99 -I./include #-save-temps
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

all: $(OBJECTS)
	$(CC) -g $(OBJECTS) -o main.out $(LDFLAGS)

.c.o:
	$(CC) -g $(CFLAGS) $< -o $@

clean:
	rm -f ./*.o ./*.i ./*.s main.out
