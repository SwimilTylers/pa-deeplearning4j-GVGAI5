package controllers.miniAlphaZero.deeplearning4j;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/7/24
 * Time: 17:33
 */
public class NetTuning {
    private static MultiLayerNetwork policy_net;
    private static MultiLayerNetwork value_net;

    private static String pnet_file = "policy";
    private static String vnet_file = "value";

    static public boolean isTuned = false;

    static public int m_maxPolicyPoolSize = 5000;
    static public int m_maxValuePolicySize = m_maxPolicyPoolSize;

    static {
        policy_net = new MultiLayerNetwork(NetConfigurations.getPolicyNetConfiguration());
        policy_net.init();

        value_net = new MultiLayerNetwork(NetConfigurations.getValueNetConfiguration());
        value_net.init();

    }

    public static MultiLayerNetwork getPolicy_net() {
        return policy_net;
    }

    public static MultiLayerNetwork getValue_net() {
        return value_net;
    }

    public static void train(DataSet pData, DataSet vData){
        System.out.println("start training net");
        policy_net.setListeners(new ScoreIterationListener(1));
        policy_net.fit(pData);
        try {
            policy_net.save(new File(pnet_file));
        } catch (IOException e) {
            e.printStackTrace();
        }

        value_net.setListeners(new ScoreIterationListener(1));
        value_net.fit(vData);
        try {
            value_net.save(new File(vnet_file));
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("finish training net");
        isTuned = true;
    }

    public static void load(){
        System.out.println("start loading net");
        try {
            policy_net = MultiLayerNetwork.load(new File(pnet_file), true);
            value_net = MultiLayerNetwork.load(new File(vnet_file), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("finish loading net");
        isTuned = true;
    }

    public static void fine_tuning(DataSet raw_pData, DataSet raw_vData){
        Random shuffle_rnd = new DefaultRandom();
        System.out.println("start fine tuning");
        DataSet pData = raw_pData.sample(m_maxPolicyPoolSize, shuffle_rnd);
        policy_net.setListeners(new ScoreIterationListener(1));
        policy_net.fit(pData);

        DataSet vData = raw_vData.sample(m_maxValuePolicySize, shuffle_rnd);
        value_net.setListeners(new ScoreIterationListener(1));
        value_net.fit(vData);
        System.out.println("finish fine tuning");
    }
}
