from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

class testModel(ABC):
    #把datdloader的所有参数存入类
    def __init__(self,**dataLoaderPars) -> None:
        super().__init__()
        self.dataLoaderPars=dataLoaderPars

    #训练AI,使用self.dataloader加载训练数据
    @abstractmethod
    def trAIn(self,dataLoader:DataLoader) ->None:
        pass
    #运行AI,输入inp,返回AI输出的结果
    @abstractmethod
    def run(self,inp:Any) ->Any:
        pass
    #把模型的输出打包成dataset作为下一次模型训练的输入
    @abstractmethod
    def outToDataset(self,inp:Any) ->Dataset:
        pass
    #获取AI每次测试需要回答的问题,n为测试已经执行的轮数
    @abstractmethod
    def getQuestion(self,n) ->Any:
        pass
    #判断AI是否崩溃
    @abstractmethod
    def hasCrashed(self) ->bool:
        pass

    #启动测试
    def test(self,dataset:Dataset) ->int:
        i=0
        crash=False
        dataLoader=DataLoader(dataset,**self.dataLoaderPars)
        while not crash:
            #训练
            self.trAIn(dataLoader)
            #推理
            out=self.run(self.getQuestion(i))
            #把生成的数据作为下一轮的训练数据
            dataLoader=DataLoader(self.outToDataset(out),**self.dataLoaderPars)
            #检测是否崩溃
            crash=self.hasCrashed()
            i+=1

        return i