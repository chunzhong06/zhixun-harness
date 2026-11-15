import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import moa.tud.ke.patching.Patching;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/** 供作者的 Weka 加载入口使用。权重输入按官方气体模型的行优先顺序导出。 */
public final class OfficialGasBase extends AbstractClassifier {
    private static final long serialVersionUID = 1L;
    private static final int INPUTS = 128, HIDDEN = 32, CLASSES = 6;
    private final double[] mean, scale, w1, b1, w2, b2;

    private OfficialGasBase(DataInputStream input) throws IOException {
        mean = read(input, INPUTS);
        scale = read(input, INPUTS);
        w1 = read(input, INPUTS * HIDDEN);
        b1 = read(input, HIDDEN);
        w2 = read(input, HIDDEN * CLASSES);
        b2 = read(input, CLASSES);
        if (input.read() != -1) throw new IOException("unexpected trailing weights");
    }

    private static double[] read(DataInputStream in, int size) throws IOException {
        double[] values = new double[size];
        for (int i = 0; i < size; i++) values[i] = in.readDouble();
        return values;
    }

    @Override public void buildClassifier(Instances data) {
        throw new UnsupportedOperationException("the competition baseline is frozen");
    }

    @Override public double classifyInstance(Instance row) {
        double[] h = b1.clone();
        for (int i = 0; i < INPUTS; i++) {
            double x = Math.max(-10, Math.min(10, (row.value(i) - mean[i]) / scale[i]));
            for (int j = 0; j < HIDDEN; j++) h[j] += x * w1[i * HIDDEN + j];
        }
        double[] logits = b2.clone();
        for (int j = 0; j < HIDDEN; j++) {
            double activation = Math.tanh(h[j]);
            for (int k = 0; k < CLASSES; k++) logits[k] += activation * w2[j * CLASSES + k];
        }
        // softmax 保持类别排序，作者接口只消费类别。
        int best = 0;
        for (int k = 1; k < CLASSES; k++) if (logits[k] > logits[best]) best = k;
        return best;
    }

    private static Instances header() {
        ArrayList<Attribute> fields = new ArrayList<>();
        for (int i = 0; i < INPUTS; i++) fields.add(new Attribute("x" + i));
        fields.add(new Attribute("gas", new ArrayList<>(Arrays.asList(
            "Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"))));
        Instances data = new Instances("official-gas", fields, 0);
        data.setClassIndex(INPUTS);
        return data;
    }

    /** 第五个参数可选，只接受官方训练集。待评价的特征文件始终没有标签。 */
    public static void main(String[] args) throws Exception {
        File model = new File(args[1]);
        try (DataInputStream in = new DataInputStream(new FileInputStream(args[0]));
             ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(model))) {
            out.writeObject(new OfficialGasBase(in));
        }
        Patching patcher = new Patching();
        patcher.initialBatchSize.setValue(1);
        boolean fitPatch = args.length == 5;
        if (fitPatch) {
            // 论文 §4.2 的二元错误区域及补丁使用随机森林；官方网络提供基础预测。
            String forest = "weka.classifiers.trees.RandomForest -I 100 -S 1";
            patcher.dsClassifierOption.setValueViaCLIString(forest);
            patcher.patchClassifierOption.setValueViaCLIString(forest);
        } else {
            patcher.forceNoAdaptation.set();
        }
        patcher.saveLoadModel.setValue(model.getAbsolutePath());
        patcher.resetLearningImpl();
        Instances schema = header();
        WekaToSamoaInstanceConverter converter = new WekaToSamoaInstanceConverter();
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(args[2])));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[3])))) {
            int n = in.readInt();
            if (n <= 0) throw new IOException("empty input");
            long predictNs = 0;
            for (int row = 0; row < n; row++) {
                double[] values = Arrays.copyOf(read(in, INPUTS), INPUTS + 1);
                values[INPUTS] = Double.NaN;
                Instance item = new DenseInstance(1.0, values);
                item.setDataset(schema);
                com.yahoo.labs.samoa.instances.Instance sample = converter.samoaInstance(item);
                // 第一条仅触发作者的模型加载入口，不提供标签，也不拟合权重。
                if (row == 0) {
                    patcher.trainOnInstanceImpl(sample);
                    if (fitPatch) {
                        Instances training = new Instances(schema, 0);
                        try (DataInputStream labeled = new DataInputStream(new BufferedInputStream(new FileInputStream(args[4])))) {
                            int count = labeled.readInt();
                            if (count <= 0) throw new IOException("empty training batch");
                            for (int i = 0; i < count; i++) {
                                double[] example = Arrays.copyOf(read(labeled, INPUTS), INPUTS + 1);
                                int label = labeled.readUnsignedByte();
                                if (label >= CLASSES) throw new IOException("invalid training label");
                                example[INPUTS] = label;
                                training.add(new DenseInstance(1.0, example));
                            }
                            if (labeled.read() != -1) throw new IOException("trailing training rows");
                        }
                        long start = System.nanoTime();
                        patcher.updateClassifier(training);
                        System.out.println("FIT_NANOSECONDS=" + (System.nanoTime() - start));
                    }
                }
                long start = System.nanoTime();
                double[] votes = patcher.getVotesForInstance(sample);
                predictNs += System.nanoTime() - start;
                if (votes.length != CLASSES || Arrays.stream(votes).sum() != 1.0)
                    throw new IOException("author predictor returned no valid class");
                int label = 0;
                while (votes[label] != 1.0) label++;
                out.writeByte(label);
            }
            if (in.read() != -1) throw new IOException("trailing rows");
            System.out.println("PREDICT_NANOSECONDS=" + predictNs);
        }
    }
}
