# 极市2020-服装风格分类-2nd 0.7292 ACC|162.4990 FPS

非常感谢极市平台提供的比赛，感谢比赛群实时给选手提供技术帮助的程序员朋友。

以下是我的一些尝试与想法,如果有不同想法或者实验结果欢迎讨论,若我有错误欢迎指出,共同学习!

## 环境

训练: pytorch python3.7

​	sklearn, torch, mmcv, opencv-python, apex, efficientnet-pytorch

推理:openvino c++

## 方案

### 1.模型

​	比赛初期首先试的是efficientnet-b0, 没做过多tricks,CV:0.85x,提交:0.75x-0.76x, 但是FPS不够理想大约是10+FPS. 后面方向就主要转到所谓的移动端模型上去了.

​	模型尝试了MobilenetV3-small(FLOPS 60多), GhostNet1.0x(FLOPS 140多), MobilenetV3-large(FLOPS 160多), ShufflenetV2.因为速度在比赛的权重还是挺高的(0.3) 所以Ghostnet1.0x和MobilenetV3-large, ShufflenetV2尝试了2-3次没有特别高的准确率出现也就没再试. 最终选择MobilenetV3-small



### 2.数据

​	因为后期C++端有速度上的BUG，我的主要实验都在前期，所以这里速度记录都是修复BUG前的速度. 模型为MobilenetV3-small.

#### 1.数据增强

```
HEIGHT = WIDTH = 224
self.transforms = Compose([
    Resize(int(HEIGHT * (256 / 224)), int(WIDTH * (256 / 224)), p=1),
    RandomCrop(height=HEIGHT, width=WIDTH, p=1),
    HorizontalFlip(p=0.5),
])
self.test_transforms = Compose([
    Resize(int(HEIGHT * (256 / 224)), int(WIDTH * (256 / 224)), p=1),
    CenterCrop(height=HEIGHT, width=WIDTH, p=1),
])
```

#### 2.数据增强2

​	Cutmix(alpha=1.0, prob=0.5) >= Mixup(alpha=1.0, prob=0.5) > GridMask(d1=10,d2=100,rotate=7,ratio=[0.4,0.6],p=0.5)

​	最终选择Cutmix

#### 3.蒸馏

1. teacher-student模式，teacher模型: efficientnet-b4>=efficientnet-b1 > seresnext50-32x4d >= densenet121

2. 自我蒸馏模式,上面实验已经表明efficientnet-b1是还可以的模型，这里没有过多尝试直接用efficientnet-b1对5折数据进行训练，然后对验证集输出oof，再使用oof和原标签按2:8混合

   [kaggle/yelan](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/discussion/154056)

   **两种模式效果差不多**

   

### Loss

​	普通softmax交叉熵 > focal-loss



### 训练

RAdam(Cosine-decay) > Adam(warm-up + Cosine-decay) 我的余弦退火是每个batch即调整学习率(每个epoch调整应该差别不大)

epochs = 20

batch_size = 256 (因为前面没有看到平台提供的GPU是什么,我没有给很大,更大可能结果会更好)

BalanceClassSampler用于平衡7个类别



### 后处理

1. 取之前最好的checkpoint，把所有数据加入训练 训练4epochs，最后平均这4个epochs的checkpoints

2. ```python
   model = torch.quantization.fuse_modules(model ,config.merge_list)
   ```

   conv-bn-relu -> convbnrelu. 提升13-15FPS



### 一些尝试但没什么用:

metric learning: Arcface loss learning

更大的input_size: 我的实验里96x96提交能到0.67x, 224x224提交能到0.72x,所以数据存在一些非常难的样本而大部分是简单样本，如果为了极限速度可以考虑96x96.(再极限点你可以32x32.......)

optimizer的选择: 尝试了over9000和Ranger效果和RAdam并无很大差别.

修改模型： 这太难了。。。也是我时间耗得最多的地方，但结果不好.

## 代码

分类代码并不复杂，所以我稍微保留了一些我的注释.

代码是按极市平台环境为基础写的 如果要本地运行需要修改路径

e4.py :  efficientnet-b1生成自我蒸馏oof

m11.py： MobilenetV3-small训练代码



openvino推理代码我没存下来。。。但是非常简单！就是处理图片然后送到网络里面没有任何tricks, 后面我能拿到 我会进行补充.

------

MobilenetV3-small is from [mobilenetV3](https://github.com/xiaolai-sqlai/mobilenetv3)

GhostNet is from [GhostNet](https://github.com/huawei-noah/ghostnet)

感谢！！