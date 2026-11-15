"""同种轴承、固定载荷工况下的历史寿命参照，使用已知运行时长预测剩余分钟。"""
from math import isfinite
from statistics import median


def number(value, allow_zero=False):
    if type(value) not in (float,int) or not isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError("寿命、载荷和转速应为正有限数；运行分钟可为零")
    return float(value)


class BearingLoadLife:
    """载荷单位 kN，转速单位 rpm。库中每条历史必须来自不同的已完成设备。"""
    def __init__(self, completed):
        products=[]
        devices=set()
        for row in completed:
            if set(row)!={"device","life_minutes","radial_load_kn","rpm"}:
                raise ValueError("历史档案需要设备身份、整段运行分钟、载荷和转速")
            device=row["device"]
            if not isinstance(device,str) or not device or device in devices:
                raise ValueError("历史设备身份无效或重复")
            devices.add(device)
            value=number(row["life_minutes"])*number(row["rpm"])*number(row["radial_load_kn"])**3
            products.append(number(value))
        if not products:
            raise ValueError("至少需要一台同型设备的已完成档案")
        self.scale=median(products)

    def predict(self, inputs):
        if not isinstance(inputs,dict) or set(inputs)!={"elapsed_minutes","radial_load_kn","rpm"}:
            raise ValueError("服务输入需要当前已运行分钟、固定径向载荷和转速")
        total=self.scale/(number(inputs["rpm"])*number(inputs["radial_load_kn"])**3)
        return max(total-number(inputs["elapsed_minutes"],allow_zero=True),0.)
