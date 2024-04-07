# cvDeeplearning
基于 SNPE 部署深度学习模型，包括目前用到和学习的各种方法，目标检测、人脸检测、跟踪、单目深度估计等。
* snpe 版本 1.68，opencv 版本 4.6
* 基于 cmake，每个方法使用单独的 CMakeLists 文件，方便单个方法的运行和部署。
* 不论什么框架部署，整体流程基本一致：加载模型（初始化）-> 图像预处理 -> 构建 tensor -> 推理 -> 后处理：
    1. &nbsp;&nbsp;&nbsp;&nbsp;加载模型（初始化）、构建 tensor 和推理一般结合部署框架和模型设置相应参数即可；
    2. &nbsp;&nbsp;&nbsp;&nbsp;预处理和后处理部分需要结合模型输入和输出进行处理；
    3. &nbsp;&nbsp;&nbsp;&nbsp;预处理通常主要包括 resize、average、normalized 等。需要结合模型训练时配置，resize 是否需要保持高宽比、是否要进行均值和归一化等处理不正确，都可能导致推理结果错误。除此之外，还要注意图像的排列方式，onnx、dnn 需要输入为 nchw，SNPE 输入时 nhwc。opencv 默认是 nhwc，通过 blob 生成 tensor 后会转变成 nchw；
    4. &nbsp;&nbsp;&nbsp;&nbsp;后处理则需要根据模型输出结果处理，不同模型的后处理一般都不一样。


# 文件目录如下：
cvDeeplearning/  
├── data/ # 测试图像数据  
├── models/ # 运行模型文件  
├── cfgs / # 模型配置文件，设置模型加载路径，输入输出等  
├── docs / # 介绍文档  
├── res / # 资源文件  
├──src/ # 源码  
│   ├── utils/ # 数据类型定义 snpe、onnx推理模块封装  
│   ├── tracker/ # 多目标跟踪相关  
│   ├── yolov8/  
│   │   ├──snpe1.68  
│   │   │   ├───Yolov8.h  
│   │   │   ├───Yolov8.cpp  
│   │   │   ├───main.cpp  
│   │   │   └── CMakeLists.txt  
│   │   ├──onnx # 目前先实现snpe，后续可能继续完成其他框架，ncnn、onnx、paddlelite等  
│   │   └─── ....  
│   ├── .....  # 其他方法定义及cmakelists  
│   ├── main.cpp  
│   └── CMakeLists.txt  
└── CMakeLists.txt # 顶层cmakelist，定义链接库目录  
# 已实现方法：  
1. 目标检测：  Yolov8  
2. 人脸检测：  Yolov8Face Retinaface  
3. 人脸关键点检测： PFLD（68点）  
4. 多目标跟踪：Deepsort Bytetrack 参考：https://github.com/shaoshengsong/DeepSORT  
5. 单目深度估计：  Midas  
6. 人体关键点： Yolov8Pose  
