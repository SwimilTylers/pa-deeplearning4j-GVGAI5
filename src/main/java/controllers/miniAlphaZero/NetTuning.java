package controllers.miniAlphaZero;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.SimpleCart;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;

import java.io.*;
import java.util.Random;

import static weka.core.Debug.DBO.pln;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/6/26
 * Time: 0:18
 */
@Deprecated
public class NetTuning {
    static private Classifier pLearner = new AdaBoostM1();
    static private Classifier vLearner = new REPTree();

    static private String pLearnerStructure = "a";
    static private String vLearnerStructure = "a";

    static public boolean isTuned = false;

    static public int m_maxPolicyPoolSize = 5000;
    static public int m_maxValuePolicySize = m_maxPolicyPoolSize;

    public static void cross_validation(Instances data, Classifier learner, int folds) throws Exception {
        Instances predictedData = null;
        Evaluation eval = new Evaluation(data);
        System.out.println("start cross validation");
        for (int i = 0; i < folds; i++) {
            Instances train = data.trainCV(folds, i);
            Instances test = data.testCV(folds, i);
            Classifier clsCopy = Classifier.makeCopy(learner);
            clsCopy.buildClassifier(train);
            eval.evaluateModel(clsCopy, test);

            // add prediction
            AddClassification filter = new AddClassification();
            filter.setClassifier(learner);
            filter.setOutputClassification(true);
            filter.setOutputDistribution(true);
            filter.setOutputErrorFlag(true);
            filter.setInputFormat(train);
            Filter.useFilter(train, filter);
            Instances pred = Filter.useFilter(test, filter);
            if (predictedData == null)
                predictedData = new Instances(pred, 0);
            for (int j = 0; j < pred.numInstances(); j++)
                predictedData.add(pred.instance(j));
        }
        pln(eval.toSummaryString("=== " + folds + " test ===", false));
    }

    public static void train(Instances pHeader, Instances vHeader, boolean CV) throws Exception {
        System.out.println("start training policy classifier");
        ((AdaBoostM1)pLearner).setClassifier(new J48());
        // ((MultilayerPerceptron)pLearner).setHiddenLayers(pLearnerStructure);
        if (CV){
            cross_validation(pHeader, pLearner, 5);
        }
        pLearner.buildClassifier(pHeader);

        System.out.println("tuned, start training value classifier");
        // ((MultilayerPerceptron)vLearner).setHiddenLayers(vLearnerStructure);
        if (CV){
            cross_validation(vHeader, vLearner, 5);
        }
        else
            vLearner.buildClassifier(vHeader);
        System.out.println("tuned, saving models");

        ObjectOutputStream poos = new ObjectOutputStream(new FileOutputStream("policy.model"));
        poos.writeObject(pLearner);
        poos.flush();
        poos.close();

        ObjectOutputStream voos = new ObjectOutputStream(new FileOutputStream("value.model"));
        voos.writeObject(pLearner);
        voos.flush();
        voos.close();
        System.out.println("end training step");
        isTuned = true;
    }

    public static void load(){
        try {
            System.out.println("start loading model");
            ObjectInputStream pois = new ObjectInputStream(new FileInputStream("policy.model"));
            ObjectInputStream vois = new ObjectInputStream(new FileInputStream("value.model"));
            pLearner = (Classifier) pois.readObject();
            vLearner = (Classifier) vois.readObject();
            pois.close();
            vois.close();
            System.out.println("end loading model");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        isTuned = true;
    }

    public static Classifier getpLearner() {
        return pLearner;
    }

    public static Classifier getvLearner() {
        return vLearner;
    }

    public static void updata_net(Instances pHeader, Instances vHeader) throws Exception {
        Random shuffle_rnd = new Random();
        System.out.println("start updating net");

        pHeader.randomize(shuffle_rnd);
        while (pHeader.numInstances() > m_maxPolicyPoolSize)
            pHeader.delete(0);

        pLearner.buildClassifier(pHeader);

        vHeader.randomize(shuffle_rnd);
        while (vHeader.numInstances() > m_maxValuePolicySize)
            vHeader.delete(0);

        vLearner.buildClassifier(vHeader);

        System.out.println("end updating net");
    }
}
