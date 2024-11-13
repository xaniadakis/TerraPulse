# Variables
CC = gcc
CFLAGS = -Wall -Iinclude
LDFLAGS = -lm -lfftw3 -lgsl -lgslcblas -lpthread
SRCDIR = src
INCDIR = include
OBJDIR = build
DAT_TARGET = $(OBJDIR)/dat_to_text
SRD_TARGET = $(OBJDIR)/srd_to_text

# Source and object files
# SOURCES = $(wildcard $(SRCDIR)/*.c)
DAT_SOURCES = $(filter-out $(SRCDIR)/p_dat_to_text.c $(SRCDIR)/srd_to_text.c, $(wildcard $(SRCDIR)/*.c))
DAT_OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(DAT_SOURCES))

SRD_SOURCES = $(filter-out $(SRCDIR)/p_dat_to_text.c $(SRCDIR)/dat_to_text.c, $(wildcard $(SRCDIR)/*.c))
SRD_OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRD_SOURCES))

# Build the dat project
dat: $(DAT_TARGET)

# Link object files into the executable
$(DAT_TARGET): $(DAT_OBJECTS)
	$(CC) $(DAT_OBJECTS) -o $@ $(LDFLAGS)

# Build the srd project
srd: $(SRD_TARGET)

# Link object files into the executable
$(SRD_TARGET): $(SRD_OBJECTS)
	$(CC) $(SRD_OBJECTS) -o $@ $(LDFLAGS)

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Debug build target
debug: CFLAGS += -g
debug: LDFLAGS += -g
debug: $(TARGET)
	@echo "Compiled with debugging symbols."

# Run the program in GDB
run_gdb_dat: debug
	gdb $(DAT_TARGET)

run_gdb_srd: debug
	gdb $(SRD_TARGET)

# Clean build files
clean:
	rm -rf $(OBJDIR)/*.o $(DAT_TARGET) $(SRD_TARGET)
