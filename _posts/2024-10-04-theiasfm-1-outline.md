---
title: 探索TheiaSfM：1-概述
description: An Overview of the TheiaSfM Project and Introductions to GFlags, GTest, and GLog
date: 2024-10-04 14:54:00 +/-0800
categories: [Computer Vision, SfM] 
tags: [SfM, TheiaSfM, GFLags, GLog, GTest]
author: stoner
image: /assets/img/computer-vision/theiasfm-1-outline.jpg
math: false
comments: true
---

## 0 从基础使用开始
先了解TheiaSfM库作为一个三维重建库的基础应用，这些使用在源码的`appliction`文件夹下有详细的举例，这里以一个基本的重建任务为例，源文件`build_reconstruction.cc`代码为：
```cpp
#include <vector>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <theia/theia.h>

using namespace theia;
int main(int argc, char* argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  CHECK_GT(FLAGS_output_reconstruction.size(), 0);

  // Initialize the features and matches database.
  std::unique_ptr<FeaturesAndMatchesDatabase> features_and_matches_database(
      new theia::RocksDbFeaturesAndMatchesDatabase(
          FLAGS_matching_working_directory));

  // Create the reconstruction builder.
  const ReconstructionBuilderOptions options =
      SetReconstructionBuilderOptions();
  ReconstructionBuilder reconstruction_builder(
      options, features_and_matches_database.get());

  // If matches are provided, load matches otherwise load images.
  if (features_and_matches_database->NumMatches() > 0) {
    AddMatchesToReconstructionBuilder(features_and_matches_database.get(),
                                      &reconstruction_builder);
  } else if (FLAGS_images.size() != 0) {
    AddImagesToReconstructionBuilder(&reconstruction_builder);
  } else {
    LOG(FATAL) << "You must specifiy either images to reconstruct or supply a "
                  "database with matches stored in it.";
  }

  std::vector<Reconstruction*> reconstructions;
  CHECK(reconstruction_builder.BuildReconstruction(&reconstructions))
      << "Could not create a reconstruction.";

  for (int i = 0; i < reconstructions.size(); i++) {
    const std::string output_file =
        theia::StringPrintf("%s-%d", FLAGS_output_reconstruction.c_str(), i);
    LOG(INFO) << "Writing reconstruction " << i << " to " << output_file;
    CHECK(theia::WriteReconstruction(*reconstructions[i], output_file))
        << "Could not write reconstruction to file.";
  }
}
```
从官方这份示例代码来看，略去`LOG`和`CHECK`日志语词，应用层上一个重建任务的构建流程十分简洁清晰，首先如果能提供特征和匹配数据库`FeaturesAndMatchesDatabase`，则将数据库传入重建类`ReconstructionBuilder`，否则将传入图像集，然后由`ReconstructionBuilder`执行重建得到保存为`std::vector<Reconsrtruction>`的重建结果，最后依次输出为文件。

当然，这里略去了繁杂的各类重建的细节参数的设置。

上述代码阅后，不难看出：

- `ReconstructionBuilder`和`Reconstruction`就是分管重建过程和重建数据的两个核心类，可以作为我们阅读源码的一个重要入口；
- 特征与匹配本身与图像分析更紧密，在上述代码中与图像集都隶属于重建的入口数据源，可以将其的阅读优先级往后放；
- 此外，我们还不禁问一个问题，为什么重建结果以`vector`的形式保存的，重建过程具体又是怎样的。

## 1 项目结构
首先一览TheiaSfM的项目结构。
```
├─applications
├─cmake
├─data
├─docs
├─include
├─libraries
└─src
    └─theia
        ├─alignment
        ├─image
        │  ├─descriptor
        │  └─keypoint_detector
        ├─io
        ├─matching
        ├─math
        │  ├─graph
        │  ├─matrix
        │  └─probability
        ├─sfm
        │  ├─bundle_adjustment
        │  ├─camera
        │  ├─estimators
        │  ├─global_pose_estimation
        │  ├─pose
        │  ├─transformation
        │  ├─triangulation
        │  └─view_graph
        ├─solvers
        ├─test
        └─util
```
`applictions`内是调用TheiaSfM库进行与三维重建相关任务的源码，`cmake`包含一些C++库查找和其他配置的CMake子模块，`data`存放一些用以测试的图像数据，`docs`是项目文档，`include`是theiaSfM库的包含路径，`libraries`是涉及到的一些三方库源码，最后`src`则是重点：TheiaSfM的源码。

- `alignment`：内含单个头文件，用以提供`Eigen`的`std::vector`特化，一遍正确处理内存对齐问题；
- `image`：图像特征提取；
- `io`：图像、特征、标定文件、平差文件、重建文件等的输入输出流管理；
- `math`：与矩阵、图和概率论相关的数学工具；
- `sfm`：涉及SfM的相机模型、位姿、三角化、重建和平差的代码核心部分；
- `solvers`：利用RANSAC及其变体进行模型估计的相关类；
- `test`：单元测试主程序，各个单元的test代码分散上各个文件夹内，以模块名辅以`_test.cc`结尾；
- `util`：一些工具函数或类，包括随机数、字符串、文件流、线程池等。

在所有的第三方库中，真正从输入开始“贯穿全文”的是GFlags、GLog和GTest，下面简单介绍这几个库的作用和使用。

## 2 GFlags简介
我们知道主函数`int main(int argc, char* argv[])`具有外部输入的参数，`argc`是外部参数数量加一，`argv`就是所有外部参数和程序字符串本身的集合。将这些参数排列成`(name, value)`对，那么生成的可执行程序`program.exe`在运行时可以有语义更加清晰的额外的配置，比如：

```shell
program -height -age 20 -name "Vincent"
```

其中`-height`、`-age`、`-name`称为命令行标志（Commandline Flags），` `、`20`、`Vincent`是对应的命令行参数（Commandline arguments)，`GFlags`就是用来为程序定义命令行标志的C++库。

`GFlags`安装[^1]很简单，官方文档[^2]很详尽，这里简洁地概括一下。

`GFlags`编译好后在CMake中配置

```cmake
find_package(gflags REQUIRED)
target_link_libraries(program gflags::gflags)
```

然后在需要使用的地方引入头文件`#include <gflags/gflags.h>`，即可使用。

如果你需要定义一个标志，使用`DEFINE_type(name, default_value, "description")`，比如

- `DEFINE_bool(is_student, false, "Is the target a student.")`
- `DEFINE_string(name, "", "Vincent")`

具体支持以下几种类型：

- `DEFINE_bool`: boolean
- `DEFINE_int32`: 32-bit integer
- `DEFINE_int64`: 64-bit integer
- `DEFINE_uint64`: unsigned 64-bit integer
- `DEFINE_double`: double
- `DEFINE_string`: C++ string

标志定义处的当前源文件内皆可访问这些标志的参数值，方式是通过标志名前面添加`FLAGS_`，比如`FLAGS_is_student`。如果要引用其它源文件内定义的标志，需要在当前源文件进行标志的前置声明：`DECLARE_bool(is_student)`。

如果需要对输出参数进行有效性检查，可以定义一个检查函数并传给`DEFINE_validator(flag_name, &validation_function)`。

```cpp
static bool IsFlagAgeValid(const char* flag_name, int value) 
{
    if (value > 0 && value < 100)
      return true;
    std::cout << "Invalid value for --" << flag_name 
              << ", value = " << value << std::endl;
   return false;
}
DEFINE_int32(age, 0, "Age of student");
DEFINE_validator(age, &IsFlagAgeValid);
```

最后在使用时需要在主函数调用解析函数：

```cpp
GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
```

取址符号表明了这个函数很可能会修改`argc`和`argv`，修改方式取决于第三个参数`remove_flags`。如果`reomve_flags`为`true`，将移除`argv`所有标志只保留参数，并调整`argc`；如果为`false`则只对`argv`进行重排，将标志移动到前面，参数移动到后面。

最后即可在调用可执行程序时，通过`program --name "Vincent"`的方式传递外部参数了，标志前单横线双横线皆可。

此外，有一些内置的默认标志。可以通过`--helpfull`显示所有的标志，使用`--flagfile`可以指定读入参数的文本文件，用来取代在命令行中手动敲入，TheiaSfM即是采用这种方式：
```
############### Input/Output ###############
# Input/output files.
# Set these if a matches file is not present. Images should be a filepath with a
# wildcard e.g., /home/my_username/my_images/*.jpg
--images= 
--output_matches_file= 
```
TheiaSfM将`GFLAGS_NAMESPACE`用宏替换为了`THEIA_GFLAGS_NAMESPACE`，但`GLog`直接用的`google::`，暂不清楚如此处理的原因。可能是处理Gflags在不同版本时的命名空间不一致的问题。

`GFlags`更多细节和注意事项请查阅官方文档。
## 3 GLog简介
`GLog`全称Google Logging Library，由Google出品的日志库。类似地在官方Github[^3]下载源码编译后由CMake引入：

```cmake
find_package(glog REQUIRED)
target_link_libraries(program glog::glog)
```

然后引入头文件并在主函数初始化即可使用`GLog`。

```cpp
#include <glog/logging.h>

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);  
}
```
`argv`即使在前文的`gFlags`初始化时进行了修改也不会造成冲突。`GLog`支持四个等级的日志输出`INFO`、`WARNING`、`ERROR`、和`FATAL`，其中`FATAL`会伴随着程序终止。

默认的日志输出格式为：

```
Lyyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg...
```

常见用法为`LOG(level) << "message"`即可打印对应等级的日志信息。

```cpp
LOG(INFO) << "Program initialized.";
```

还可以附带条件地打印日志信息：

```cpp
LOG_IF(INFO, is_success) << "Program initialized.";
```

我们可以为日志设置额外的等级，通过`VLOG(level)`进行输出，经由命令行标志`-v`来进行等级控制，只有小于设置等级的`VLOG`被打印。

```cpp
VLOG(1) << "I’m printed when you run the program with --v=1 or higher";
VLOG(2) << "I’m printed when you run the program with --v=2 or higher";
```

`GLog`还提供`CHECK`宏进行运行时条件检查，不满足时直接终止程序。

```cpp
CHECK(name == "Vincent") << "Error person!";
CHECK_NE(1, 2) << ": The world must be ending!";
CHECK_EQ(string("abc")[1], 'b') << "That's strange!";
CHECK_NOTNULL(some_ptr) << "Empty pointer!";
```

以上就是`GLog`的常见用法，更多细节以及和`GFlags`搭配使用请参考官方文档[^4]。
## 4 GTest简介
类似的，`GTest`从Github官方[^5]下载源码编译后，在`CMake`中引入
```cmake
find_package(GTest)
target_link_libraries(program GTest::gtest GTest::gtest_main)
```

`GTest`通过写各种断言来对目标条件进行检测，失败后当前函数终止，否则程序继续执行。一条测试用例通过`TEST(TestUnitName, TestName)`定义，内部包含诸多断言语句。`ASSERT_`断言会产生致命失败并结束当前函数，`EXPECT_`断言会产生非致命失败但不终止当前函数。

```cpp
// Tests factorial of 0.
TEST(FactorialTest, HandlesZeroInput) {
  EXPECT_EQ(Factorial(0), 1);
}

// Tests factorial of positive numbers.
TEST(FactorialTest, HandlesPositiveInput) {
  EXPECT_EQ(Factorial(1), 1);
  EXPECT_EQ(Factorial(2), 2);
  EXPECT_EQ(Factorial(3), 6);
  EXPECT_EQ(Factorial(8), 40320);
}
```
在主函数种初始化后再`RUN_ALL_TESTS()`即可执行所有测试。
```cpp
int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
```

执行目标程序时即可运行相应的测试用例

```shell
program # run all tests
program --gtest_filter=* # run all tests
program --gtest_filter=TargetUnit.* # run TestUnitName == TargetUnit
program --gtest_filter=TargetUnit.*-TargetUnit.Outsider # run TestUnitName == TargetUnit, except for TargetUnit.Outsider。
```
其它与`GLog`相关的更为复杂的设置参加官方文档[^6]。

## 5 小结
本文概述了TheiaSfM的项目结构以及简单介绍了`GFlags`、`GLog`和`GTest`的使用。

## 参考
[^1]: GFlags Github. [https://github.com/gflags/gflags](https://github.com/gflags/gflags)
[^2]: How To Use gflags (formerly Google Commandline Flags). [https://gflags.github.io/gflags](https://gflags.github.io/gflags/)
[^3]: GLog Github. [https://github.com/google/glog](https://github.com/google/glog)
[^4]: Google Logging Library[https://google.github.io/glog](https://google.github.io/glog)
[^5]: GTest Github. [https://github.com/google/googletest](https://github.com/google/googletest)
[^6]: GoogleTest User’s Guide. [https://google.github.io/googletest](https://google.github.io/googletest)