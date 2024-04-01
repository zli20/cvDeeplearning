# cvDeeplearning
基于snpe部署深度学习模型  
snpe版本 1.68 opencv版本 4.6  
使用cmake  
每个方法使用单独的cmakelists文件，方便单个方法的运行和部署  
文件目录如下：  
cvDeeplearning/  
├── data/ # 测试图像数据  
├── models/ # 运行模型文件  
└── src/ # 源码  
│   ├── utils/ # 数据类型定义 snpe、onnx推理模块封装  
│   ├── tracker/ # 多目标跟踪相关  
│   ├── yolov8/  
│   │   ├──Yolov8.h  
│   │   ├──Yolov8.cpp  
│   │   └── CMakeLists.txt  
│   ├── .....  # 其他各类方法定义及cmakelists  
│   ├── main.cpp  
│   └── CMakeLists.txt  
└── CMakeLists.txt # 顶层cmakelist，定义链接库目录  
已实现方法：  
1. 目标检测：  Yolov8
2. 人脸检测：  Yolov8Face
3. 人脸关键点检测： PFLD（56点）
4. 多目标跟踪：Deepsort Bytetrack  参考：https://github.com/shaoshengsong/DeepSORT  
5. 单目深度估计：  
6. 人体关键点： Yolov8Pose
