# Variables
CC = gcc
CFLAGS = -Wall -Iinclude
LDFLAGS = -lm -lfftw3 -lgsl -lgslcblas -lpthread
SRCDIR = src
INCDIR = include
OBJDIR = build
TARGET = $(OBJDIR)/p_dat_to_text

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

# Debug build target
debug: CFLAGS += -g
debug: LDFLAGS += -g
debug: $(TARGET)
	@echo "Compiled with debugging symbols."

# Run the program in GDB
run_gdb: debug
	gdb $(TARGET)

# Clean build files
clean:
	rm -rf $(OBJDIR)/*.o $(TARGET)
