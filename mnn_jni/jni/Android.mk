LOCAL_PATH := $(call my-dir)

OpenCV_BASE = ../3rdlib/opencv-4.1
MNN_BASE    = ../3rdlib/mnn
NET_BASE = ./net_engine

MODULE_BASE = ./alg

include $(CLEAR_VARS)
LOCAL_MODULE := MNN
LOCAL_SRC_FILES := $(MNN_BASE)/libs/arm64-v8a/libMNN.so
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS)

OpenCV_INSTALL_MODULES := on
OPENCV_LIB_TYPE := STATIC
include $(OpenCV_BASE)/sdk/native/jni/OpenCV.mk

$(warning "opencv include dir $(OPENCV_INCLUDE_DIR)")
LOCAL_C_INCLUDES += $(OPENCV_INCLUDE_DIR)
LOCAL_C_INCLUDES += $(MNN_BASE)/include
LOCAL_C_INCLUDES += $(NET_BASE)
LOCAL_C_INCLUDES += $(MODULE_BASE)/include

#$(MODULE_BASE)/src/mtcnn.cpp
#$(MODULE_BASE)/src/Bbox.cpp
LOCAL_SRC_FILES := 	$(NET_BASE)/net.cpp\
					$(MODULE_BASE)/src/UltraLightFastGenericGaceDetector1MB.cpp\
					$(MODULE_BASE)/src/imgProcess.cpp


LOCAL_LDLIBS := -landroid -llog -ldl -lz 
LOCAL_CFLAGS   := -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -ftree-vectorize -fPIC -Ofast -ffast-math -w -std=c++14
LOCAL_CPPFLAGS := -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fPIC -Ofast -ffast-math -std=c++14
LOCAL_LDFLAGS  += -Wl,--gc-sections
LOCAL_CFLAGS   += -fopenmp
LOCAL_CPPFLAGS += -fopenmp
LOCAL_LDFLAGS  += -fopenmp
LOCAL_ARM_NEON := true

APP_ALLOW_MISSING_DEPS = true

LOCAL_SHARED_LIBRARIES :=                             \
                        MNN
						
LOCAL_MODULE     := DetectMNN

include $(BUILD_SHARED_LIBRARY)
