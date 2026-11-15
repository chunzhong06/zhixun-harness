"""执行模型生成的纯训练代码：限制可用导入和I/O能力，保留源码与发布权重身份。"""
import ast
import builtins
import hashlib
from pathlib import Path
import sys
import types


IMPORTS={
    "numpy":None,"math":None,
    "sklearn.base":{"BaseEstimator","RegressorMixin","TransformerMixin","clone"},
    "sklearn.pipeline":{"Pipeline","make_pipeline"},
    "sklearn.preprocessing":{"StandardScaler","RobustScaler","PolynomialFeatures","FunctionTransformer"},
    "sklearn.impute":{"SimpleImputer"},
    "sklearn.compose":{"ColumnTransformer","TransformedTargetRegressor"},
    "sklearn.linear_model":{"Ridge","LinearRegression","HuberRegressor","ElasticNet","BayesianRidge","LogisticRegression","RidgeClassifier"},
    "sklearn.ensemble":{"ExtraTreesRegressor","RandomForestRegressor","HistGradientBoostingRegressor","VotingRegressor",
                        "ExtraTreesClassifier","RandomForestClassifier","HistGradientBoostingClassifier"},
    "sklearn.neighbors":{"KNeighborsRegressor"},"sklearn.svm":{"SVR"},
    "sklearn.kernel_ridge":{"KernelRidge"},
}
NUMPY={"array","asarray","column_stack","hstack","vstack","concatenate","stack","zeros","ones",
       "zeros_like","ones_like","empty","arange","maximum","minimum","clip","where","isfinite",
       "nan_to_num","log1p","log","exp","sqrt","square","abs","sign","sin","cos","tanh",
       "mean","median","std","sum","min","max","quantile","percentile","average","cov",
       "pi","nan","float64","float32","int64","ndarray","linalg","newaxis","c_","r_",
       "power","divide","multiply","subtract","add","isnan","isinf","finfo","atleast_2d"}
FORBIDDEN={"open","exec","eval","compile","input","globals","locals","vars","getattr","setattr",
           "delattr","dir","help","breakpoint","exit","quit","load","save","dump","tofile",
           "fromfile","memmap","ctypes","ctypeslib","os","sys","subprocess","socket","pathlib"}


def inspect_source(source):
    if len(source.encode())>20000:raise ValueError("训练代码超过20KB")
    tree=ast.parse(source);numpy_names=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Lambda):raise ValueError("发布对象不能引用lambda，请定义模块顶层函数或类")
        if isinstance(node,ast.Import):
            for alias in node.names:
                if alias.name not in ("numpy","math"):raise ValueError("仅允许列出的纯计算库")
                if alias.name=="numpy":numpy_names.add(alias.asname or alias.name)
        if isinstance(node,ast.ImportFrom):
            if node.level or node.module not in IMPORTS:raise ValueError("不允许该导入")
            allowed=NUMPY if node.module=="numpy" else IMPORTS[node.module]
            if any(a.name=="*" or (allowed is not None and a.name not in allowed) for a in node.names):
                raise ValueError("导入对象不属于已开放接口")
        if isinstance(node,ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN):
            raise ValueError("训练代码不得读取外部状态或调用动态执行")
        if isinstance(node,ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN:raise ValueError("不允许私有属性或I/O操作")
            if isinstance(node.value,ast.Name) and node.value.id in numpy_names and node.attr not in NUMPY:
                raise ValueError("该numpy操作不在纯计算接口中")
        if isinstance(node,(ast.Global,ast.Nonlocal)):raise ValueError("模型状态应保存在拟合对象中")
    if not any(isinstance(n,ast.FunctionDef) and n.name=="build_model" for n in tree.body):
        raise ValueError("需定义无参数build_model()并返回可fit/predict的对象")
    return tree


def load_module(path):
    source=Path(path).read_text(encoding="utf-8");tree=inspect_source(source)
    identity="field_regressor_"+hashlib.sha256(source.encode()).hexdigest()
    module=types.ModuleType(identity);module.__file__=str(path)
    safe={name:getattr(builtins,name) for name in ("__build_class__","object","super","property","staticmethod",
          "classmethod","len","range","enumerate","zip","list","tuple","dict","set","float","int","bool",
          "str","min","max","sum","abs","round","sorted","reversed","isinstance","issubclass","type",
          "ValueError","TypeError","RuntimeError","Exception")}
    def allowed_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level or name not in IMPORTS:raise ImportError("不允许该模块")
        return builtins.__import__(name,globals,locals,fromlist,level)
    safe["__import__"]=allowed_import;module.__dict__["__builtins__"]=safe
    sys.modules[identity]=module
    exec(compile(tree,str(path),"exec"),module.__dict__)
    return module


def check_model(model):
    if not all(callable(getattr(model,name,None)) for name in ("fit","predict","get_params")):
        raise ValueError("模型需要fit、predict和get_params")
    for key,value in model.get_params(deep=True).items():
        name=key.rsplit("__",1)[-1]
        if name=="n_jobs" and value not in (None,1):raise ValueError("模型须使用单线程")
        if name in ("n_estimators","max_iter") and value is not None and value>(128 if name=="n_estimators" else 256):
            raise ValueError("树至多128棵，迭代至多256次")
        if name=="memory" and value is not None:raise ValueError("模型不能另建磁盘缓存")
