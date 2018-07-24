package controllers.miniAlphaZero.pretraining;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/6/25
 * Time: 19:08
 */
public class InstancesStore {
    static public Instances pHeader = pDataExtractor.datasetHeader();
    static public Instances vHeader = vDataExtractor.datasetHeader();
    static public boolean isStored = false;

    static public void save(){
        Random shuffle_rnd = new Random();

        System.out.println("start saving data");
        ArffSaver psaver = new ArffSaver();
        pHeader.randomize(shuffle_rnd);
        psaver.setInstances(pHeader);
        try {
            psaver.setFile(new File("pData.arff"));
            psaver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }

        ArffSaver vsaver = new ArffSaver();
        vHeader.randomize(shuffle_rnd);
        vsaver.setInstances(vHeader);
        try {
            vsaver.setFile(new File("vData.arff"));
            vsaver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("end saving data");
        isStored = true;
    }

    static public void load(){
        Random shuffle_rnd = new Random();

        System.out.println("start loading data");

        ArffLoader ploader = new ArffLoader();
        try {
            ploader.setFile(new File("pData.arff"));
            pHeader = ploader.getDataSet();
            pHeader.setClassIndex(pHeader.numAttributes()-1);
            pHeader.randomize(shuffle_rnd);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ArffLoader vloader = new ArffLoader();
        try {
            vloader.setFile(new File("vData.arff"));
            vHeader = vloader.getDataSet();
            vHeader.setClassIndex(vHeader.numAttributes()-1);
            vHeader.randomize(shuffle_rnd);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("end loading data");
        isStored = true;
    }
}
