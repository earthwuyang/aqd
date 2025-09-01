# ============================================================
#  router – 单 Makefile 构建脚本（2025-06-25 double-fix 版）
# ============================================================

# ── ① 基础参数 ───────────────────────────────────────────────
CXX            ?= g++
CXX_STD        ?= c++17
OPTFLAGS       ?= -O3
WARNFLAGS      ?= -Wall -Wextra -Wshadow -Wno-sign-compare

USE_EXP_FS     ?= 1
WITH_LIGHTGBM  ?= 1
WITH_FANN      ?= 1
WITH_MYSQL     ?= 1
WITH_OPENMP    ?= 1

# ── ② 第三方前缀 ─────────────────────────────────────────────
JSON_ROOT      ?= /home/wuy/software/json-develop/single_include/nlohmann
LIGHTGBM_ROOT  ?= /home/wuy/software/LightGBM
FANN_ROOT      ?= /usr
MYSQL_ROOT     ?= /usr

# ── ③ 源文件 ────────────────────────────────────────────────
SRCS  := main.cpp lightgbm_model.cpp fannmlp_model.cpp \
         decision_tree_model.cpp gin_model.cpp \
         global_stats.cpp common.cpp
OBJS  := $(SRCS:.cpp=.o)
TARGET := router

# ============================================================
# ④ 自动探测 json.hpp
# ============================================================
ifeq ($(wildcard $(JSON_ROOT)/json.hpp),)
  JSON_ROOT := $(firstword \
      $(wildcard /home/wuy/software/json-develop/single_include/nlohmann) \
      /usr/local/include /usr/include)
endif
ifeq ($(wildcard $(JSON_ROOT)/json.hpp),)
  $(error "✖ json.hpp 未找到：请指定 JSON_ROOT=/path/to/nlohmann")
endif

# ============================================================
# ⑤ 编译 / 链接基本选项
# ============================================================
CXXFLAGS = $(OPTFLAGS) $(WARNFLAGS) -std=$(CXX_STD) -pipe \
           -I$(JSON_ROOT) -I$(JSON_ROOT)/nlohmann -I/usr/include/eigen3
LDFLAGS  =
LDLIBS   = -pthread

ifeq ($(USE_EXP_FS),1)
  CXXFLAGS += -DUSE_EXP_FS
  LDLIBS   += -lstdc++fs
endif

# OpenMP
OPENMP_STATUS = NO
ifeq ($(WITH_OPENMP),1)
  ifeq ($(shell $(CXX) -fopenmp -dM -E -x c++ /dev/null >/dev/null 2>&1 && echo yes),yes)
    CXXFLAGS += -fopenmp -DUSE_OPENMP
    OPENMP_STATUS = YES
  else
    $(warning "⚠ 编译器不支持 OpenMP，已禁用")
  endif
endif

# ============================================================
# ⑥ 可选依赖：LightGBM / FANN / MySQL
# ============================================================

# 工具函数：检测“头 AND 库”都存在
define both_exist
$(and $(wildcard $(1)),$(wildcard $(2)))
endef

# ---- LightGBM ----
LGBM_STATUS = NO
ifeq ($(WITH_LIGHTGBM),1)
  LGBM_HDR := $(LIGHTGBM_ROOT)/include/LightGBM/config.h
  LGBM_SO  := $(firstword $(wildcard $(LIGHTGBM_ROOT)/lib_lightgbm.*))
  ifneq ($(call both_exist,$(LGBM_HDR),$(LGBM_SO)),)
    CXXFLAGS += -I$(LIGHTGBM_ROOT)/include
    LGBM_DIR := $(dir $(LGBM_SO))
    ifneq ($(strip $(LGBM_DIR)),)
      LDFLAGS += -L$(LGBM_DIR)
    endif
    LDLIBS   += -l_lightgbm
    CXXFLAGS += -DWITH_LIGHTGBM
    LGBM_STATUS = YES
  else
    $(warning "⚠ LightGBM 头或库缺失 – 已禁用")
  endif
endif

# ---- FANN ----
FANN_STATUS = NO
ifeq ($(WITH_FANN),1)
  FANN_HDR := $(FANN_ROOT)/include/fann.h
  FANN_SO  := $(firstword $(wildcard $(FANN_ROOT)/lib*/libfann.*))
  ifneq ($(call both_exist,$(FANN_HDR),$(FANN_SO)),)
    CXXFLAGS += -I$(FANN_ROOT)/include
    FANN_DIR := $(dir $(FANN_SO))
    ifneq ($(strip $(FANN_DIR)),)
      LDFLAGS += -L$(FANN_DIR)
    endif
    LDLIBS   += -lfann
    CXXFLAGS += -DWITH_FANN
    FANN_STATUS = YES
  else
    $(warning "⚠ FANN 头或库缺失 – 已禁用")
  endif
endif

# ---- MySQL ----
MYSQL_STATUS = NO
ifeq ($(WITH_MYSQL),1)
  MYSQL_HDR := $(firstword \
      $(wildcard $(MYSQL_ROOT)/include/mysql/mysql.h) \
      $(wildcard $(MYSQL_ROOT)/include/mysql.h))
  MYSQL_SO  := $(firstword $(wildcard $(MYSQL_ROOT)/lib*/mysql/libmysqlclient.*))
  ifneq ($(call both_exist,$(MYSQL_HDR),$(MYSQL_SO)),)
    CXXFLAGS += -I$(dir $(MYSQL_HDR))
    MYSQL_DIR := $(dir $(MYSQL_SO))
    ifneq ($(strip $(MYSQL_DIR)),)
      LDFLAGS += -L$(MYSQL_DIR)
    endif
    LDLIBS   += -lmysqlclient
    MYSQL_STATUS = YES
  else
    $(warning "⚠ MySQL 头或库缺失 – 已禁用")
  endif
endif

# ============================================================
# ⑦ 构建规则
# ============================================================
.PHONY: all clean summary

all: $(TARGET) summary

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

summary:
	@echo "--------------------------------------------------"
	@echo " Build summary"
	@echo "   C++ standard : $(CXX_STD)"
	@echo "   JSON root    : $(JSON_ROOT)"
	@echo "   LightGBM     : $(LGBM_STATUS)"
	@echo "   FANN         : $(FANN_STATUS)"
	@echo "   MySQL        : $(MYSQL_STATUS)"
	@echo "   OpenMP       : $(OPENMP_STATUS)"
	@echo "   Exp. FS      : $(if $(USE_EXP_FS),YES,NO)"
	@echo "   LDFLAGS      : $(LDFLAGS)"
	@echo "   LDLIBS       : $(LDLIBS)"
	@echo "   Target       : $(TARGET)"
	@echo "--------------------------------------------------"
