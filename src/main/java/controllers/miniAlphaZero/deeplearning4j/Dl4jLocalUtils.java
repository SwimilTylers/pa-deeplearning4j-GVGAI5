package controllers.miniAlphaZero.deeplearning4j;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/7/24
 * Time: 15:57
 */
public class Dl4jLocalUtils {
    public static DataSet getDataSetFromInstances(Instances original){
        INDArray array = null;
        INDArray label = null;

        int total_feature = original.numAttributes();
        int label_index = original.classIndex();
        int total_instances = original.numInstances();

        for (int i = 0; i < total_feature; i++) {
            if (i == label_index) {
                label = Nd4j.create(original.attributeToDoubleArray(i));
                label = label.reshape(total_instances, 1);
            }
            else {
                INDArray buffer = Nd4j.create(original.attributeToDoubleArray(i), new int[]{total_instances, 1});
                if (array == null)
                    array = buffer;
                else
                    array = Nd4j.hstack(array, buffer);
            }
        }

        assert array != null;
        array = array.reshape(total_instances, total_feature-1);

        return new DataSet(array, label);
    }

    public static INDArrayDataSetIterator getINDArrayDataSetIteratorFromInstances(Instances original, int batch_size){
        INDArray array = null;
        INDArray label = null;

        int total_feature = original.numAttributes();
        int label_index = original.classIndex();
        int total_instances = original.numInstances();

        for (int i = 0; i < total_feature; i++) {
            if (i == label_index) {
                label = Nd4j.create(original.attributeToDoubleArray(i));
                label = label.reshape(total_instances, 1);
            }
            else {
                INDArray buffer = Nd4j.create(original.attributeToDoubleArray(i), new int[]{total_instances, 1});
                if (array == null)
                    array = buffer;
                else
                    array = Nd4j.hstack(array, buffer);
            }
        }

        assert array != null;
        array = array.reshape(total_instances, total_feature-1);
        final INDArray instances = array;
        final INDArray marks = label;

        return new INDArrayDataSetIterator(new Iterable<Pair<INDArray, INDArray>>() {
            @NotNull
            @Override
            public Iterator<Pair<INDArray, INDArray>> iterator() {
                return new Iterator<Pair<INDArray, INDArray>>() {
                    private int current_iter = 0;
                    private int total_count = instances.rows();
                    @Override
                    public boolean hasNext() {
                        return current_iter < total_count;
                    }

                    @Override
                    public Pair<INDArray, INDArray> next() {
                        current_iter++;
                        return new Pair<>(instances.getRow(current_iter-1), marks.getRow(current_iter-1));
                    }
                };
            }
        }, batch_size);
    }

    public static void main(String[] args) {
        Instances info = null;
        ArffLoader ploader = new ArffLoader();
        try {
            ploader.setFile(new File("test.arff"));
            info = ploader.getDataSet();
            info.setClassIndex(info.numAttributes()-1);
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataSet ds = getDataSetFromInstances(info);
        System.currentTimeMillis();
    }
}
