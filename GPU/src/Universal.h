#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H

#define PINNED 1
//PINNED = 0 -- no pinned memory
//PINNED = 1 -- pinned memory on output buffer

#define CALLBACK 1
//CALLBACK = 0 -- force no callback
//CALLBACK = 1 -- force use callback unless WIN64 is defined

#define CONCURRENTKERNELCPY 1
//CONCURRENTKERNELCPY = 0 Not concurrent. Separate scale and copy functions
//CONCURRENTKERNELCPY = 1 Scale kernel then copy, 4 streams
//CONCURRENTKERNELCPY = 2 Scale kernel 4x then copy 4x
//#define SPEEDTEST 1
//#define DEBUG 1
//#define debug_print(fmt, ...) \
        do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)


#define _DEBUG
#ifdef _DEBUG
#define Print(s)  fprintf(stderr, "%s", s)
#else
#define Print(s)  
#endif

#endif