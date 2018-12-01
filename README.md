# AIliuys
### 第十周作业

- 数据集准备
  - 数据集准备都是在线下完成，修改代码详见https://github.com/liuysong/AIliuys/blob/master/20180919_week10/quiz-w9-code/convert_fcn_dataset.py
  - 在处理训练数据时和校验数据集时，
    - 训练集中以下图片和相应的标定图片，图片size小于244x224做丢弃处理:    
      2007_002273.jpg, 2008_004055.jpg, 2008_005294.jpg, 2009_000720.jpg, 2009_001898.jpg, 2009_004626.jpg, 2009_005055.jpg, 2010_002935.jpg
    - 校验集中以下图片size小于244x224做丢弃处理:    
      2007_003848.jpg, 2007_004468.jpg, 2007_008802.jpg, 2007_009096.jpg, 2008_000666.jpg, 2008_000673.jpg, 2008_003108.jpg, 2008_004175.jpg, 2009_000487.jpg, 2010_000216.jpg, 2010_002939.jpg, 2010_005187.jpg, 2011_001722.jpg
  - 数据集处理生成文件fcn_train.record,fcn_val.record，详见https://www.tinymind.com/Louis/datasets/w9-tfrecord
- 模型代码补全:
  - 修改train.py，详见代码https://github.com/liuysong/AIliuys/blob/master/20180919_week10/quiz-w9-code/train.py
  - 在原有代码基础上
    - 添加了8s节点处网络的导出代码，以获取8s特征图8s特征图
    - 将原本16s节点处，反向卷积后并联后做的16被反向卷积修改为步长为2的卷积，即特征图会扩大2x2倍
    - 在8s处，将原有8s节点处的特征图和上述两次反卷积后得到的8s大小的特征图拼接后，再次进行步长为8的反向卷积，特征图总共扩大2x2x8倍，即达到和原图同样大小。
- 训练过程
  - 训练日志详见: https://www.tinymind.com/executions/5fevjqbi
  - 由于训练参数batchsize 相比于作业要求的16缩减了2倍，所以本次训练的step也相应的增加2倍，即batchsize=8，maxstep= 3000
- 结果输出
  - 运行开始后，会在400step后，每隔200个step输出模型对校验集的校验结果，详见: https://www.tinymind.com/executions/5fevjqbi/output/eval
  - 其中参考val_3000_prediction.jpg，可以看出对标定的动物有明显的紫色标记，同时可以看出在原图车轮胎和左上角天空处，有其他颜色的标记，虽然目前看来没有明显的准确意义，这个应该是模型对其他物体的识别
  - 其中的val_3000_prediction_crfed.jpg，物品分割框架经过CRF后期处理后的结果，可以明显的看出对标定物体的分割形状
  - 观察运行过程中的loss分布，损失一直处于在抖动中下降的趋势，所以认为模型的准确率尚未达到上限，经过更多次的训练，应该可以获得更好的结果
