# Variables
CC = gcc
CFLAGS = -Wall -Iinclude
LDFLAGS = -lm -lfftw3 -lgsl -lgslcblas
SRCDIR = src
INCDIR = include
OBJDIR = build
TARGET = $(OBJDIR)/dat_to_text

# Source and object files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES))

# Build the project
all: $(TARGET)

# Link object files into the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJDIR)/*.o $(TARGET)
