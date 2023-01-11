# Dataset Inference代码使用简单指南

## Ⅰ、安装 

- 安装环境： Python 3.8
- 安装依赖： pip install -r requirements.txt

## Ⅱ、目录结构

``` c
Dataset-Inference
|-- README.md
|-- data #数据集存放文件夹
|   |-- cifar-10-batches-py
|   |-- cifar-10-python.tar.gz
|   `-- ti_500K_pseudo_labeled.pickle  #!!不会自动下载需要手动下载，下载地址: https://drive.google.com/file/d/1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi/view
|-- files # 训练置信模型用到的feature文件，并非模型（虽然可以改feature存放位置但不建议）
|   `-- CIFAR10 #数据集名称
|       `-- model_extract-label_normalized #model_id名称
|           |-- test_rand_vulnerability_2.pt
|           `-- train_rand_vulnerability_2.pt
|-- models # 训练好的模型存放根目录
|   `-- CIFAR10 #数据集名
|       |-- model_extract-label_normalized #model_id名称
|       |   |-- final.pt
|       |   |-- logs.txt
|       |   `-- model_info.txt
|       |-- model_extract_normalized
|       |   |-- final.pt
|       |   |-- logs.txt
|       |   `-- model_info.txt
|       |-- model_fine-tune_normalized
|       |   |-- final.pt
|       |   |-- logs.txt
|       |   `-- model_info.txt
|       |-- model_fine-tune_unnormalized
|       |   |-- final.pt
|       |   |-- logs.txt
|       |   `-- model_info.txt
|       |-- model_fine-tuning_test
|       |   |-- logs.txt
|       |   `-- model_info.txt
|       |-- model_teacher_normalized
|       |   |-- final.pt
|       |   |-- iter_24.pt
|       |   |-- iter_49.pt
|       |   |-- logs.txt
|       |   `-- model_info.txt
|       `-- model_test
|           |-- final.pt
|           |-- iter_24.pt
|           |-- iter_49.pt
|           |-- logs.txt
|           `-- model_info.txt
|-- requirements.txt #安装用到的文件
|-- src
|   |-- __pycache__
|   |-- attacks.py
|   |-- funcs.py
|   |-- generate_features.py
|   |-- model_src
|   |   |-- cnn.py
|   |   |-- preactresnet.py
|   |   |-- resnet.py
|   |   |-- resnet_8x.py
|   |   `-- wideresnet.py
|   |-- models.py #!resnet34网络的读取原作者未完成，最好不要使用resnet34
|   |-- notebooks # .py文件负责训练模型、计算feature，最后计算P_value和画图交给notebook完成
|   |   |-- CIFAR100_mingd.ipynb
|   |   |-- CIFAR100_rand.ipynb
|   |   |-- CIFAR10_mingd.ipynb
|   |   |-- CIFAR10_rand.ipynb 
|   |   |-- SVHN_rand.ipynb
|   |   `-- utils.py
|   |-- params.py
|   `-- train.py
`-- ti_500K_pseudo_labeled.pickle #这里也需要复制一份
```

​	在models和file下的生成的模型文件和feature都将按照**（根——数据集名称——自定义名称——xxx.pt）**的路径存放。值得一提的是无标注数据集文件（如CIFAR10对应了`ti_500K_pseudo_labeled.pickle`需要自行补充。**无标注数据集的作用是作为模型窃取攻击时补充训练使用。因此在对原模型进行攻击时需要注明：`--pseudo_labels 1`**）

## Ⅲ、 训练模型

​		文件夹`src`下 `train.py`负责训练和攻击，可以使用以下类似代码进行训练和攻击

``` bash
python train.py --batch_size 1000 --mode $MODE --normalize $NORMALIZE --model_id $MODEL_ID --lr_mode $LR_MODE --epochs $EPOCHS --dataset $DATASET --lr_max $LR_MAX --pseudo_labels $PSEUDO
```

但这种方法并不推荐，因为容易忘记如何设置的训练方式，不过对于此问题作者设置了模型文件夹下的`log.txt`文件记录训练相关信息

​		更推荐修改`params.py`的参数，里面有详尽的参数解释。而在提取的feature和计算p_value的那一步里，也同样使用此params.py的参数设计。这里介绍一些重要的参数

- --dataset 数据集选择 choices = ["ImageNet","MNIST", "SVHN", "CIFAR10", "CIFAR100","AFAD"] 用于选择数据集

  - #! TODO: **缺少自定义数据集**

- --model_type, help="cnn/wrn-40-2/wrn-28-10/preactresnet/resnet34" 

  - **这个没啥用，在代码中有 net_mapper = {"CIFAR10":WideResNet, "CIFAR100":WideResNet, "AFAD":resnet34, "SVHN":ResNet_8x}写明了指定数据集使用特定的网络结构。**
  - **resnet34原作者未完成，不要用**

- --batch_size 一个batch有多少，不必多说

- --model_id 自定义名称（用作目录）

  - 会在输入的model_id的自定义名称前加上`“model_”`作为前缀

- --normalize , 是否正则化初始数据

- --device，一般是0，使用老师的服务器看情况改

- --epochs， 不必多说

- --mode， help = "Various threat models", type = *str*, default = 'extract-label', choices = ['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher','independent','pre-act-18','random']

  - **本词条涉及到加载模型使用的模型深度尤其在extract_label、fine-tuning等攻击的时候不仅要在train的时候改，在计算feature的时候也需要更改**

- "--pseudo_labels" 

  - 使用另一个数据集，**在进行`extract_labels`，`extract-logit`，`fine-tuning`的时候作为偷窃模型后偷窃者自身的数据集之用**
  - DI中使用的是 [**"ti_500K_pseudo_labeled.pickle"**]( https://drive.google.com/file/d/1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi/view)

- "--feature_type", help = "Feature type for generation", type = *str*, default = '', choices = ['pgd','topgd', 'mingd', 'rand'])

  - 生成feature使用的攻击方法

-   --lr_mode", help = "Step wise or Cyclic", type = *int*, default=1

    --opt_type", help = "Optimizer", type = *str*, default = "SGD"

    --lr_max", help = "Max LR", type = *float*, default = 0.1

    --lr_min", help = "Min LR", type = *float*, default = 0.

  - 以上是学习率等超参数的设置。**其中opt只有两种选择 选择SDG，如果不是SDG则会选择Adam**，学习率相关的设置如下：

    ``` python
    def step_lr(lr_max, epoch, num_epochs):
        ratio = epoch/float(num_epochs)
        if ratio < 0.3: return lr_max
        elif ratio < 0.6: return lr_max*0.2
        elif ratio <0.8: return lr_max*0.2*0.2
        else: return lr_max*0.2*0.2*0.2
    
    def lr_scheduler(args):
        if args.lr_mode == 0:
            lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [args.lr_max, args.lr_max*0.2, args.lr_max*0.04, args.lr_max*0.008])[0]
        elif args.lr_mode == 1:
            lr_schedule = lambda t: np.interp([t], [0, args.epochs//2, args.epochs], [args.lr_min, args.lr_max, args.lr_min])[0]
        elif args.lr_mode == 2:
            lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [args.lr_min, args.lr_max, args.lr_max/10, args.lr_min])[0]
        elif args.lr_mode == 3:
            lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [args.lr_max, args.lr_max, args.lr_max/5., args.lr_max/10.])[0]
        elif args.lr_mode == 4:
            lr_schedule = lambda t: step_lr(args.lr_max, t, args.epochs)
        return lr_schedule
    ```

    共四种模式可选。

- ……待补充



**训练模型时要注意：**

1. 训练受害者模型（待攻击）时模型自定义id不要乱取，要叫`teacher_normalized`或`teacher_unnormalized`，否则需要修改源码`train.py`的167行和188行的`teacher_dir`的值。以及计算p_value的时候的源码
2. 学习率4种选择都比较奇特，尤其是2，随着训练epoch越高learning_rate越来越大（可能是我设置错误？）

## Ⅳ、 计算feature

​	**！最好是**每训练完一个模型就用feature_extract计算这个模型的feature，否则要改很多params。model_id在train里是存放的目录，而在extract这一步是读取模型文件的目录

​	**！有些代码需要修改**

​		遇到`RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation.`

```python
remaining[remaining] = new_remaining #把报错的那一行
#更改为
remaining_temp = remaining.clone()
remaining[remaining_temp] = new_remaining`
```

## Ⅴ、 计算p_value

计算p_value只能用他给的notebook，这里最好使用思宇的DI代码加载计算出的feature.pt进行计算。notebook代码的做论文比较有用，但不适合实验初步阶段使用。
