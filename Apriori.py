import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import dm2022exp

class Apriori(object):

    def __init__(self, min_sup:float=0.2, min_confidence:float = 0.2):
        self.min_sup = min_sup
        self.min_confidence = min_confidence
        pass

    def Apriori_k(self, Dataset, Frequent_K, k, Len_X, Fre_dict, Dict_B): #生成k项频繁集
        Frequent_temp = [] #储存k项候选集
        for i in range(len(Frequent_K) - 1):
            for j in range(i + 1, len(Frequent_K)):
                temp = []
                if k == 2:
                    temp = Frequent_K[i] + Frequent_K[j]
                else:
                    Count = 0 #记录是不是前k - 2 项都相等（因为Frequent_K只存储了k - 1项数据）
                    for z in range(k - 2):
                        if Frequent_K[i][z] != Frequent_K[j][z]:
                            Count = 1
                            break
                        # if z == k - 3:
                    if Count == 0:
                        temp += Frequent_K[i]
                        temp.append(Frequent_K[j][k - 2])
                if temp:
                    Frequent_temp.append(temp) #生成候选集

        Frequent_K = [] #用于储存此次的k项集，因此先置为空集
        fre_k_change = dict() #存储当前k项集的支持度的字典
        for element_F in Frequent_temp: #筛选候选集，生成频繁集
            a = 0
            for dataset in Dataset:
                Count = 0
                if len(dataset) >= k:
                    for i in range(k):
                        if dataset.count(element_F[i]):
                            Count += 1
                        else:
                            Count = 0
                            break
                if Count > 0:
                    a = a + 1
            a = a / Len_X
            if a >= self.min_sup:
                Frequent_K.append(element_F)
                fre_change = self.decode_fre(element_F, Dict_B)
                fre_k_change[fre_change] = a
                Fre_dict[fre_change] = a
        return Frequent_K, Fre_dict, fre_k_change
        pass

    def apriori_one(self, Dataset, Len_X, Fre_dict, Dict_B): #寻找1项频繁集
        Frequent_temp = []
        Frequent_K = []
        Frequent_set = []
        Support_degree = []
        for i in Dataset:
            for j in i:
                Frequent_temp.append(j) #生成候选项

        Frequent_temp = list(set(Frequent_temp))

        for value in Frequent_temp: #筛选频繁项
            a = 0
            for element in Dataset:
                a += element.count(value)  #统计该候选项出现频率
            a = a / Len_X
            if a >= self.min_sup: #如果出现频率大于等于最低频率，说明是频繁项
                f_K = [value]
                Frequent_K.append(f_K)#将频繁k项集记录
                Frequent_set.append(f_K) #将生成的频繁k项添加当频繁项集里面去
                Support_degree.append(a)  #将生成的频繁项的支持度放入
                fre_change = self.decode_fre(f_K, Dict_B) #将该项频繁集解码
                Fre_dict[fre_change] = a #将解码后的频繁集转为不变集合放入字典
        #print(len(Frequent_set))
        return Frequent_K, Frequent_set, Support_degree, Fre_dict

    def Record_Key(self,B): #记录数字对应的元素
        Dict_A = dict() #记录X中的元素所替换的数字
        Dict_B = dict() #记录数字所对应的x中的元素
        Number_replace = 0 #用于替换的数字
        for element_B in B:
            Dict_A[element_B] = Number_replace
            Dict_B[Number_replace] = element_B
            Number_replace = Number_replace + 1
        return Dict_A, Dict_B

    def decode_fre(self, fre_list, Dict_B):
        fre_change = []
        for value in fre_list:
            temp = Dict_B[value]
            fre_change.append(temp)
        fre_change = frozenset(fre_change)
        return fre_change

    def confidence_rule(self, Fre_rule, fre_k_change, fre_dict): #用于求取关联规则里的置信度
        '''
        :param Fre_rule: 存储关联规则的置信度
        :param fre_k_change: 当前k项集
        :param fre_dict: 所有频繁集
        :return:
        '''
        for value in fre_k_change.keys():
            element_list = self.get_sub_set(list(value)) #该频繁集的所有子集的列表
            element_support = fre_k_change[value] #该频繁集的支持度
            for element in element_list: #遍历每一个子集，求取每个关联规则的置信度
                element_set = frozenset(element) #将该子集转变为不变集合
                no_element_set = value.difference(element_set) #获得该集合里除了该子集元素的子集
                element_set_sup = fre_dict[element_set] #获得该子集的支持度
                confidence = element_support / element_set_sup #获得该关联规则的置信度
                if confidence >= self.min_confidence: #如果置信度大于设置的最小置信度，则将该关联规则放入字典
                    element_set_temp = set(element_set)
                    no_element_set_temp = set(no_element_set)
                    Fre_rule[str(element_set_temp) + ' --> ' + str(no_element_set_temp)] = confidence
        return Fre_rule
        pass

    def get_sub_set(self, nums):
        """
        给定一个列表，返回一个含有该列表所有子集的列表
        :param nums:
        :return:
        """
        sub_sets = [[]]
        for x in nums:
            sub_sets.extend([item + [x] for item in sub_sets])
            pass
        sub_sets.remove([])
        sub_sets.remove(nums)
        return sub_sets

    def fit(self, X):
        A = [] #储存X里面出现过的所有元素
        for Element_x in X:
            for element in Element_x:
                A.append(element)

        B = list(set(A)) #将A中重复的元素去除
        #B.sort()
        # print(B)
        Dict_A, Dict_B = self.Record_Key(B) #A记录X中的元素所替换的数字,B记录数字所对应的x中的元素

        Dataset = [] #存储转变为数字后的x元素
        for i in X:
            temp = []
            for j in i:
                temp.append(Dict_A[j])
            Dataset.append(temp)

        Len_X = len(X) #数据集x的项集数目
        #k储存k项频繁集,set储存所有的频繁项集,Sup储存所有频繁项集的支持度

        Fre_dict = dict() #记录频繁集以及支持度
        Frequent_K, Frequent_set, Support_degree, Fre_dict = self.apriori_one(Dataset, Len_X, Fre_dict, Dict_B)
        k = 1 #确定当前是k项集

        Fre_Rule = dict() #用字典来记录关联规则，关键字为x-->y，值为置信度

        while len(Frequent_K) > 1: #生成k项频繁集

            k = k + 1
            Frequent_K, Fre_dict, fre_k_change = \
                self.Apriori_k(Dataset, Frequent_K, k, Len_X, Fre_dict, Dict_B) #寻找k项频繁集
            #print(k,len(Frequent_set))

            #下面函数用于求取频繁集的置信度：
            Fre_Rule = self.confidence_rule(Fre_Rule,fre_k_change, Fre_dict)


        res = []
        for value in Fre_dict.keys():
            res.append([value, Fre_dict[value]])

        return res, Fre_Rule

if __name__ == "__main__":
    data = dm2022exp.load_ex5_data()
    m = Apriori(0.005, 0.55)
    res, fre_rule = m.fit(data)
    m = TransactionEncoder()
    m.fit(data)
    m.transform(data)
    df = pd.DataFrame(m.transform(data), columns=m.columns_)
    ret = apriori(df, min_support=0.005, use_colnames=True, verbose=True)
    #---------------------------------------------------------------------------
    # print(res)
    # print(ret) # print的内容为频繁项集和其支持度，其中ret是库函数的支持度和频繁集，res是自己的
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    for value in fre_rule.keys():
        print(value , " : ", fre_rule[value]) #打印关联规则置信度
    print("\n\n")

    rule = association_rules(ret, metric="confidence", min_threshold=0.55) #调用库函数
    #输出库函数的关联规则，与自己的比对
    print(rule)
    print(len(fre_rule))
    #--------------------------------------------------------------------------


