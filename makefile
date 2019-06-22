#------------------------
# makefile for gromov_wasser
#------------------------
SRC_DIR = src
INCL_DIR_OPT = 

# LIBS = -lm -lopenblas
LIBS = -lm -lcblas
LIBS_DIR_OPT = 

LIBRARY = libsb_descriptors.a
LIBRARY_DIR = /usr/local/lib
HEADERS_DIR = /usr/local/include/sb_desc
EXAMPLE = sb_example

#------------------------
# Source code for project
#------------------------
SRCS = sb_desc.c matrix.c sbessel.c sbesselz.c utility.c vector.c 
EXAMPLE_SRCS = example.c

#------------------------
# Compiler and linker options
#------------------------
CC   	    = gcc
FLAGS     = -std=c11 -O3
DBG_FLAGS = -g -Wall -Wextra -Wfatal-errors

OBJS = $(patsubst %.c, $(SRC_DIR)/%.o, $(SRCS))
DBG_OBJS = $(patsubst %.c, $(SRC_DIR)/%.do, $(SRCS))
EXAMPLE_OBJS = $(patsubst %.c, $(SRC_DIR)/%.o, $(EXAMPLE_SRCS))

HEADERS = $(wildcard $(SRC_DIR)/sb_*.h)

LIBS += $(patsubst lib%.a, -l%, $(LIBRARY))
LIBS_DIR_OPT += -L$(LIBRARY_DIR)
LIB_PATH = $(LIBRARY_DIR)/$(LIBRARY)

.PHONY: default debug install uninstall example docs clean

#---------------------------
# make
#---------------------------
default : $(OBJS)
	rm -f $(LIBRARY); ar rcs $(LIBRARY) $(OBJS)

#---------------------------
# make debug
#---------------------------
debug : $(DBG_OBJS)
	rm -f $(LIBRARY); ar rcs $(LIBRARY) $(DBG_OBJS)

#---------------------------
# make install
#---------------------------
install :
	@mkdir -p $(LIBRARY_DIR) && mkdir -p $(HEADERS_DIR);                         \
	if [ -f $(LIBRARY) ]; then                                                   \
		if cp $(LIBRARY) $(LIBRARY_DIR) && cp $(SRC_DIR)/sb_*.h $(HEADERS_DIR); then \
			echo "Installed library to $(LIBRARY_DIR) and headers to $(HEADERS_DIR)";\
		else                                                                       \
			echo "Installation failed. Maybe you should try 'sudo make install'?";   \
		fi                                                                         \
	else                                                                         \
		echo "Please build the library with 'make' or 'make debug' first";         \
	fi

#---------------------------
# make uninstall
#---------------------------
uninstall :
	@if rm -f $(LIB_PATH) $(HEADERS_DIR)/sb_*.h; then                            \
		echo "Uninstalled library in $(LIBRARY_DIR) and headers in $(HEADERS_DIR)";\
	else                                                                         \
		echo "Uninstallation failed. Maybe you should try 'sudo make uninstall'?"; \
	fi

#---------------------------
# make example
#---------------------------
example : INCL_DIR_OPT += -I$(HEADERS_DIR)
example : $(EXAMPLE_OBJS) $(LIB_PATH)
	$(CC) $(FLAGS) $(EXAMPLE_OBJS) -o $(EXAMPLE) $(LIBS_DIR_OPT) $(LIBS)

#---------------------------
# make docs
#---------------------------
docs :
	@doxygen src/doxyfile;                       \
	if [ ! -f "docs.html" ];                     \
		then ln -s docs/html/index.html docs.html; \
	fi

#---------------------------
# make requirements
#---------------------------
%.o : %.c
	$(CC) -c $(FLAGS) $(INCL_DIR_OPT) $< -o $@

%.do : %.c
	$(CC) -c $(DBG_FLAGS) $(INCL_DIR_OPT) $< -o $@

#---------------------------
# make clean
#---------------------------
clean :
	rm -f $(OBJS) $(DBG_OBJS) $(EXAMPLE_OBJS) $(LIBRARY) $(EXAMPLE) docs.html
