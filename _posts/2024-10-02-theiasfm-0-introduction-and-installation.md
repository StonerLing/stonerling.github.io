---
title: 探索TheiaSfM：0-介绍与安装
date: 2024-10-02 16:20:00 +/-0800
categories: [Computer Vision, SfM] 
tags: [SfM, TheiaSfM]
author: stoner
image: /assets/img/computer-vision/theiasfm-0-introduction-and-installation.jpg
math: false
comments: true
---
## 1 TheiaSfM简介

[TheiaSfM](http://www.theia-sfm.org/index.html) 是由 Chris Sweeney 开发的一款专注于结构从运动（Structure from Motion，简称 SfM）任务的计算机视觉库。它以可靠性、易用性和扩展性为核心特点，实现了包括多视图几何、位姿求解、特征提取与匹配、三维重建等关键算法。TheiaSfM 的代码风格清晰，外部依赖性低，这使得它在处理大规模 SfM 任务时表现出了优异的性能。项目的源代码已在 [Github](https://github.com/sweeneychris/Theia) 上开源，并且提供了详尽的 [API 文档](http://www.theia-sfm.org/api.html)。

## 2 TheiaSfM 安装

与其他一些知名的三维重建库如 OpenMVG 和 Colmap 相比，TheiaSfM 的文档和资料相对较少[^1]。TheiaSfM 的实际依赖库不多，其中 glog、gflags 和 cereal 已经包含在项目内部，外部依赖仅有 OpenImageIO、Eigen 和 Ceres。Eigen 库只需引入头文件即可，Ceres 库的编译也相对简单。较为复杂的是 OpenImageIO 库的编译，因为它的版本众多，而网上的博客确定能够与 TheiaSfM 兼容的 OpenImageIO 版本是 1.7.17[^2]。此外，OpenImageIO 的依赖 OpenEXR 在 Windows 环境下的编译也相当具有挑战性。

### 2.1 方法一：使用 vcpkg 安装

你可以通过 vcpkg 直接安装 TheiaSfM 库：

```shell
vcpkg install theia:[triplet]
```

其中，`[triplet]` 根据你的编译器和库类型而定，例如 `x64-windows-dynamic` 或 `x64-mingw-static`。

### 2.2 方法二：从源码编译

如果你希望从源码进行调试，可以考虑先安装 TheiaSfM 的依赖，然后从源码编译。在实际编译过程中，你会发现 TheiaSfM 的外部依赖并不算多，如果使用源码自带的库和 CMakeLists 文件，编译过程会相对轻松。

对于各个依赖库，常规编译即可。较为复杂的是 SuiteSparse，在 Windows 下 FindBlas 经常报错，这时你需要在 SuiteSparse 的 CMakeLists 文件中手动指定相应的库目录：

```cmake
set(BLAS_LIBRARIES YOUR_BLAS_DIR)
set(LAPACK_LIBRARIES YOUR_LAPACK_DIR)
```

此外，SuiteSparse 库还需要 gmp 和 mpfr 库，建议使用 vcpkg 进行安装，安装 mpfr 时会自动安装 gmp：

```shell
vcpkg install mpfr
```

配置好源码环境中的依赖项后，你可以按照常规方式进行编译和运行。

- OpenImageIO命名空间：`oiio::`->`OIIO::`
- `getline`函数在Windows平台没有实现
  
```cpp
// Choose implementation of getline according to platform
#ifdef _WIN32
    #define PLATFORM_WINDOWS
#elif __linux__ || __APPLE__
    #define PLATFORM_POSIX
#else
    #error "Unknown platform"
#endif

#ifdef PLATFORM_WINDOWS
    #include <malloc.h>
    ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
        char *bufptr = NULL;
        size_t size = 0;
        int c;

        if (lineptr == NULL || n == NULL) return -1;

        bufptr = *lineptr;
        size = *n;

        c = fgetc(stream);
        while (c != EOF && c != '\n') {
            if (bufptr - *lineptr < size - 1) {
                *bufptr++ = c;
            }
            c = fgetc(stream);
        }

        *bufptr = '\0';
        if (bufptr - *lineptr >= size - 1) {
            size_t new_size = size * 2;
            char *new_bufptr = (char *)realloc(*lineptr, new_size);

            if (new_bufptr == NULL) return -1;

            *lineptr = new_bufptr;
            *n = new_size;
            bufptr = *lineptr + size;
            while (c != EOF && c != '\n') {
                if (bufptr - *lineptr < *n - 1) {
                    *bufptr++ = c;
                }
                c = fgetc(stream);
            }
            *bufptr = '\0';
        }

        return bufptr - *lineptr;
    }
#else
    #include <stdio.h>
#endif
```

## 参考
[^1]: 库安装：TheiaSfM. [https://zhuanlan.zhihu.com/p/54112018](https://zhuanlan.zhihu.com/p/54112018)
[^2]: theia-sfm配置文档. [https://zhuanlan.zhihu.com/p/38880542](https://zhuanlan.zhihu.com/p/38880542)