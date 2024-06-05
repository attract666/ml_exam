## 配置环境

```
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git

conda create -y -n dassl python=3.8

conda activate dassl

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

cd Dassl.pytorch/

pip install -r requirements.txt

python setup.py develop
```

安装好之后可以通过`import dassl`来检查一下是否配置完成


## 下载数据集

请根据[dataset.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)下载数据集。目前仅支持`caltech101`

也可以直接通过下面的git命令下载整个项目，里面包含`caltech101`数据集

```
git clone https://github.com/attract666/ml_exam.git
```

将下载好的数据集按以下目录保存

```
ml-exam/
|-- data/
|   |-- caltech-101
|   |   |--101_ObjectCategories/
|   |   |--split_zhou_Caltech101.json
|-- plot-coop/
```

## 运行脚本
运行脚本在`/plot-coop/scripts/main.sh`，请注意修改脚本中的`DATA`为数据集根目录，以及`DIR`中的工作路径

最后运行命令
```
cd /ml_exam/plot-coop/scripts/
bash ./main.sh {dataset_name} {N}
```

其中`{dataset_name}`为数据集名称，例如`caltech101`

`{N}`为提示词数量，例如`4`
