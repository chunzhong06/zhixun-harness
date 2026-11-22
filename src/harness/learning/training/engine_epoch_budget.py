"""为结构可静态计数的 C 提案提前发现必然超额的父网训练。"""
import ast


def check_engine_epochs(source, maximum):
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"]
    if len(funcs) != 1 or len(funcs[0].args.args) != 1:
        return
    func = funcs[0]
    session = func.args.args[0].arg
    if any(isinstance(n, (ast.Return, ast.Break, ast.Continue, ast.Try, ast.While, ast.If, ast.With))
           or isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in (session,"range")
           for n in ast.walk(func)):
        return

    def direct(stmt):
        expr = stmt.value if isinstance(stmt, ast.Expr) else None
        if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name) and expr.func.value.id == session):
            return 0
        if expr.func.attr == "train_epoch":
            return 1
        if expr.func.attr == "build_regression_components" and not any(k.arg is None for k in expr.keywords):
            count = next((k.value for k in expr.keywords if k.arg == "epochs"), ast.Constant(60))
            if isinstance(count, ast.Constant) and type(count.value) is int:
                return max(0, count.value)
        return 0

    epochs = 0
    for stmt in func.body:
        epochs += direct(stmt)
        if isinstance(stmt, ast.For):
            call = stmt.iter
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"
                    and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                    and type(call.args[0].value) is int):
                epochs += max(0, call.args[0].value) * sum(direct(item) for item in stmt.body)
    if epochs > maximum:
        raise ValueError(f"C 提案至少请求 {epochs} 个父网 epoch，本次上限 {maximum}；请缩短训练程序")
