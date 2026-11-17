"""容器命令分派；一次平台调用只执行一轮，离线回放需显式请求。"""
import sys

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if '--train' in args:
        from harness.adapters.platforms.platform_train import main as train
        return train(args)
    if '--checkpoint' in args:
        from harness.adapters.platforms.baselines.predict import main as predict
        return predict(args)
    if args and args[0]=='replay':
        from harness.cli.research import main as replay
        return replay(args[1:])
    if args in ([],['--help'],['-h']):
        print('训练: harness-contest --profile A|B|C --train DATA --validation DATA --resume PARENT --output DIR')
        print('预测: harness-contest --checkpoint MODEL < inputs.jsonl')
        print('回放: harness-contest replay TASK --output DIR [研究选项]')
        return 0
    raise ValueError('未知命令；使用 --help 查看训练、预测和离线回放入口')

if __name__=='__main__':raise SystemExit(main())
