package controllers.miniAlphaZero.deeplearning4j;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;

import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/7/24
 * Time: 16:45
 */
public class NetConfigurations {
    private static MultiLayerConfiguration pnet_conf;
    private static MultiLayerConfiguration vnet_conf;

    static {
        int pInSize = 872, pOutSize = 4;
        int vInSize = 872, vOutSize = 1;

        pnet_conf = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .updater(new Adam())
                .weightInit(WeightInit.NORMAL)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nIn(pInSize)
                        .nOut(500)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nOut(500)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(pOutSize)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.feedForward(pInSize))
                .backprop(true).pretrain(false).build();


        vnet_conf = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .updater(new Adam())
                .weightInit(WeightInit.NORMAL)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nIn(vInSize)
                        .nOut(500)
                        .dropOut(0.5)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nOut(500)
                        .dropOut(0.5)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nOut(vOutSize)
                        .activation(Activation.IDENTITY)
                        .build())
                .setInputType(InputType.feedForward(vInSize))
                .backprop(true).pretrain(false).build();
    }

    public static MultiLayerConfiguration getPolicyNetConfiguration(){
        return pnet_conf;
    }

    public static MultiLayerConfiguration getValueNetConfiguration(){
        return vnet_conf;
    }
}
