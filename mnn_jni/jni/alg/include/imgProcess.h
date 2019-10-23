#pragma once

#define DLL_PUBLIC __attribute__ ((visibility("default")))

#ifdef __cplusplus
extern "C"
{
#endif
	
DLL_PUBLIC int deal(int* pixels, int h, int w, const char* path);

#ifdef __cplusplus
}
#endif
