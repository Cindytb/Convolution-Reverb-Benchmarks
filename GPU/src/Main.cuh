#pragma once
#ifndef _MAIN_H_
#define _MAIN_H


#include "Universal.h"
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include "Convolution.cuh"
#include "MemManage.cuh"
#include "Audio.cuh"

void printMe(passable *p);
float *gpuEntry(std::string input, std::string reverb, std::string out, bool timeDomain);
void drawGraph(float * buf, long long size, int SR, const char * name);
#endif
