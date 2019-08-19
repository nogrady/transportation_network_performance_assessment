package dizhuang.transportation_network_performance;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.CombinatoricsUtils;
import org.jblas.DoubleMatrix;

import brandeis.compressiveprivacy.libsvm.svm_predict;
import brandeis.compressiveprivacy.libsvm.svm_train;
import smile.regression.RandomForest;

public class efficiency_test {
	public static String alg="_FW_0.02_0.10_2.00_flow";
	public static int nLinksets=1000;
	public static int max_links=3;
	
	public static int[] nLinksets_set= {1000};
	public static int[] max_links_set= {3};
	public static String[] ntwks= {"SiouxFalls", "Tiergarten", "Anaheim", "Berlin3c"};
	
	public static double Quantile=0.45;
	public static double[] Quantile_set= {0.45};
	public static double Quantile_value;
	public static double Quantile_count_test_high;
	public static double Quantile_count_test_low;
	private static double trainDataRatio=0.8;
	private static DoubleMatrix trainData;
	private static DoubleMatrix testData;
	private static int[] trainlabel_svm;
	private static int[] testlabel_svm;
	private static double[] traintarget_regression;
	private static double[] testtarget_regression;
	
	private static int[] traintarget_rank;
	private static int[] testtarget_rank;
	
	public static int vals_high_indx_te_cnt;
	
	private static DoubleMatrix trainData_ensemble;
	private static DoubleMatrix testData_ensemble;
	private static int[] trainlabel_svm_ensemble;
	private static int[] testlabel_svm_ensemble;
	private static double[] traintarget_regression_ensemble;
	private static double[] testtarget_regression_ensemble;
	
	private static int[] traintarget_rank_ensemble;
	private static int[] testtarget_rank_ensemble;
	
	
	public static int ListSize=10;
	
	public static int[] Feats_set= {1};
	
	public static String rand_dir="RankLib-master/bin/";
	public static int rand_itrs=1;
	
	public static HashMap<Integer, Integer> ORG2RDM;
	
	private static double thres_cv=0.85;
	private static int folder_cv=2;
	
	public static int ensemble_cnt=1;
	
	private static double[] param_c={Math.pow(2, -8),  Math.pow(2, -6), 
			Math.pow(2, -4), Math.pow(2, -2), Math.pow(2, 0), 
			Math.pow(2, 2),Math.pow(2, 4), Math.pow(2, 6), Math.pow(2, 8)};
	private static double[] param_g_ml={Math.pow(2, -8),  Math.pow(2, -6), 
			Math.pow(2, -4), Math.pow(2, -2), Math.pow(2, 0), 
			Math.pow(2, 2),Math.pow(2, 4), Math.pow(2, 6), Math.pow(2, 8)};
	
	private static int SizeOfThreadPool=13;
	
	public static int ntrees=128;
	
	public static void under_sampling() throws NumberFormatException, IOException {
		HashMap<Integer, ArrayList<Integer>> label_idxs=new HashMap<Integer, ArrayList<Integer>>();
		for(int i=0;i<trainlabel_svm.length;i++) {
			if(label_idxs.containsKey(trainlabel_svm[i])) {
				label_idxs.get(trainlabel_svm[i]).add(i);
			}
			else {
				ArrayList<Integer> tmp=new ArrayList<Integer>();
				tmp.add(i);
				label_idxs.put(trainlabel_svm[i], tmp);
			}
		}
		
		ArrayList<Integer> label_0=new ArrayList<Integer>(label_idxs.get(0));
		ArrayList<Integer> label_1=new ArrayList<Integer>(label_idxs.get(1));
		
		Collections.shuffle(label_0);
		
		trainData_ensemble=new DoubleMatrix(label_1.size()*2, trainData.columns);
		trainlabel_svm_ensemble=new int[label_1.size()*2];
		traintarget_regression_ensemble=new double[label_1.size()*2];
		
		for(int i=0;i<label_1.size();i++) {
			trainlabel_svm_ensemble[i]=trainlabel_svm[label_1.get(i)];
			trainData_ensemble.putRow(i, trainData.getRow(label_1.get(i)));
			traintarget_regression_ensemble[i]=traintarget_regression[label_1.get(i)];
			
			trainlabel_svm_ensemble[i+label_1.size()]=trainlabel_svm[label_0.get(i)];
			trainData_ensemble.putRow(i+label_1.size(), trainData.getRow(label_0.get(i)));
			traintarget_regression_ensemble[i+label_1.size()]=traintarget_regression[label_0.get(i)];
		}
		
		/****************************************************************************/
		
		label_idxs=new HashMap<Integer, ArrayList<Integer>>();
		for(int i=0;i<testlabel_svm.length;i++) {
			if(label_idxs.containsKey(testlabel_svm[i])) {
				label_idxs.get(testlabel_svm[i]).add(i);
			}
			else {
				ArrayList<Integer> tmp=new ArrayList<Integer>();
				tmp.add(i);
				label_idxs.put(testlabel_svm[i], tmp);
			}
		}
		
		label_0=new ArrayList<Integer>(label_idxs.get(0));
		label_1=new ArrayList<Integer>(label_idxs.get(1));
		
		Collections.shuffle(label_0);
		
		testData_ensemble=new DoubleMatrix(label_1.size()*2, testData.columns);
		testlabel_svm_ensemble=new int[label_1.size()*2];
		testtarget_regression_ensemble=new double[label_1.size()*2];
		
		for(int i=0;i<label_1.size();i++) {
			testlabel_svm_ensemble[i]=testlabel_svm[label_1.get(i)];
			testData_ensemble.putRow(i, testData.getRow(label_1.get(i)));
			testtarget_regression_ensemble[i]=testtarget_regression[label_1.get(i)];
			
			testlabel_svm_ensemble[i+label_1.size()]=testlabel_svm[label_0.get(i)];
			testData_ensemble.putRow(i+label_1.size(), testData.getRow(label_0.get(i)));
			testtarget_regression_ensemble[i+label_1.size()]=testtarget_regression[label_0.get(i)];
		}
		Quantile_count_test_high=label_1.size();
		Quantile_count_test_low=label_1.size();
		rankScaling();
	}
	
	public static int[] ids= {2, 3, 4, 5};
	
	public static void main( String[] args ) throws IOException, InterruptedException{
		
		for(int it=0;it<100;it++) {
			for(int i:nLinksets_set) {
				nLinksets=i;
				for(int j:max_links_set) {
					max_links=j;
					for(String k:ntwks) {
						long t1=System.currentTimeMillis();
						dataGeneration(k, 1);
						long t2=System.currentTimeMillis();
						
						PrintWriter pw2 = new PrintWriter(new FileOutputStream(new File("sim_time"), true)); 
						pw2.println(k+"\t"+1+"\t"+(t2-t1));
						pw2.close();
					}
				}
			}
			
			for(double Q: Quantile_set) {
				Quantile=Q;
				
				ensemble_cnt=(int) (Math.ceil((1.0-Quantile)/Quantile));//(int) (Math.ceil((1.0-Quantile)/Quantile)> 5 ? Math.ceil((1.0-Quantile)/Quantile) : 5);
				rand_itrs=(int) (Math.ceil((1.0-Quantile)/Quantile));//(int) (Math.ceil((1.0-Quantile)/Quantile)> 5 ? Math.ceil((1.0-Quantile)/Quantile) : 5);
				
				for(int i=3;i<=3;i++) {
					for(int j: Feats_set) {
						exp("SiouxFalls", j, i);
						exp("Tiergarten", j, i);
						exp("Anaheim", j, i);
						exp("Berlin3c", j, i);
					}
				}
			}
			
			for(int i:nLinksets_set) {
				nLinksets=i;
				for(int j:max_links_set) {
					max_links=j;
					for(String k:ntwks) {
						
						for(int id:ids) {
							long t1=System.currentTimeMillis();
							dataGeneration(k, id);
							long t2=System.currentTimeMillis();
							
							PrintWriter pw2 = new PrintWriter(new FileOutputStream(new File("sim_time"), true)); 
							pw2.println(k+"\t"+id+"\t"+(t2-t1));
							pw2.close();
						}
						
						
					}
				}
			}
			
			
		}
		
		
	}
	
	public static void exp(String ntwk, int num, int num_links) throws IOException, NumberFormatException, InterruptedException {
		dataShuffling(ntwk, num, num_links);
		classification_ensemble(ntwk, num, num_links);
 		regression_ensemble(ntwk, num, num_links);
 		ranking_ensemble(ntwk, num, num_links);
//		for(int i: ListSize_set) {
//			ListSize=i;
//			ranking_ensemble(ntwk, num, num_links);
//		}
	}
	
	public static void regression_ensemble(String ntwk, int num, int num_links) throws NumberFormatException, IOException {
		String prefix=ntwk+"\t"+num+"\t"+num_links+"\t"+Quantile+"\t"+ntrees;
		int[] testlabel_0=new int[testlabel_svm.length];
		int[] testlabel_1=new int[testlabel_svm.length];
		
		long tim=0;
		
		for(int k=0;k<ensemble_cnt;k++) {
			under_sampling();
			DoubleMatrix xtrainData=new DoubleMatrix(trainData_ensemble.rows, trainData_ensemble.columns);
			xtrainData.copy(trainData_ensemble);
			DoubleMatrix xtestData=new DoubleMatrix(testData.rows, testData.columns);
			xtestData.copy(testData);
			
			double[][] xtr=xtrainData.toArray2();		
			RandomForest rf=new RandomForest(xtr, traintarget_regression_ensemble, ntrees);

			for(int i=0;i<xtestData.rows;i++) {
				long t1=System.currentTimeMillis();
				double predict_res=rf.predict(xtestData.getRow(i).toArray());
				long t2=System.currentTimeMillis();
				
				tim+=t2-t1;
				
				if(predict_res<Quantile_value) {
					testlabel_0[i]++;
				}
				else {
					testlabel_1[i]++;
				}
			}
		}
		
		PrintWriter pw=new PrintWriter(new FileOutputStream(new File("time_regression"), true)); 
		pw.println(ntwk+"\t"+tim);
		pw.close();
		
		int[] predicted=new int[testlabel_svm.length];
		for(int i=0;i<testlabel_0.length;i++) {
			if(testlabel_0[i]>=testlabel_1[i]) {
				predicted[i]=0;
			}
			else {
				predicted[i]=1;
			}
		}
		
		F1Score_regression(predicted, testlabel_svm, prefix);
	}
		
	public static void classification_ensemble(String ntwk, int num, int num_links) throws IOException {
		String prefix=ntwk+"\t"+num+"\t"+num_links+"\t"+Quantile;
		int[] testlabel_0=new int[testlabel_svm.length];
		int[] testlabel_1=new int[testlabel_svm.length];
		
		long tim=0;
		
		for(int i=0;i<ensemble_cnt;i++) {
			under_sampling();
			DoubleMatrix xtrainData=new DoubleMatrix(trainData_ensemble.rows, trainData_ensemble.columns);
			xtrainData.copy(trainData_ensemble);
			DoubleMatrix xtestData=new DoubleMatrix(testData.rows, testData.columns);
			xtestData.copy(testData);
			tim+=run_svm(xtrainData, xtestData, trainlabel_svm_ensemble, testlabel_svm, "results_svm_ensemble");

			
			
			BufferedReader br=new BufferedReader(new FileReader("results_svm_ensemble"));
			String line="";
			int cnt=0;
		    while ((line=br.readLine()) != null){
		    	if((int)Double.parseDouble(line)==0) {
		    		testlabel_0[cnt]++;
		    	}
		    	else {
		    		testlabel_1[cnt]++;
		    	}
		    	cnt++;
		    }
		    br.close();
		}
		
		PrintWriter pw=new PrintWriter(new FileOutputStream(new File("time_classification"), true)); 
		pw.println(ntwk+"\t"+tim);
		pw.close();
		
		
		(new File("results_svm_ensemble")).delete();
		
		PrintWriter pwx=new PrintWriter("results_svm");
		for(int i=0;i<testlabel_0.length;i++) {
			if(testlabel_0[i]>=testlabel_1[i]) {
				pwx.println(0);
			}
			else {
				pwx.println(1);
			}
		}
		pwx.close();
		
		
		String timeStamp = new SimpleDateFormat("yyyy/MM/dd-HH:mm:ss").format(new Date());
		pwx=new PrintWriter(new FileOutputStream("classification_ensemble_results.txt", true));
		pwx.println(timeStamp+"\t"+prefix+F1Score_classification("results_svm"));
		pwx.close();
		
		(new File("results_svm")).delete();
	}
	
	public static void ranking_ensemble(String ntwk, int num, int num_links) throws NumberFormatException, IOException, InterruptedException {
		String prefix=ntwk+"\t"+num+"\t"+num_links+"\t"+Quantile+"\t"+ListSize+"\t"+rand_itrs;
		int[] testlabel_0=new int[vals_high_indx_te_cnt*2];
		int[] testlabel_1=new int[vals_high_indx_te_cnt*2];
		
		long tim=0;
		
		for(int k=0;k<ensemble_cnt;k++) {
			under_sampling();
			
			LinkedHashMap<Integer, Integer> index_hit_cnt=new LinkedHashMap<Integer, Integer>();
			HashMap<Integer, Integer> index_hit_cnt_sorted=new HashMap<Integer, Integer>();
			int index_hit_cnt_split_value=-1;
			
			
			boolean flag=true;
			
			int itrs_flag=0;
			
			while(flag && (rand_itrs+itrs_flag-1)<=5) {
				int rand_itrs_ins=1;
				if(itrs_flag==0)
					rand_itrs_ins=rand_itrs;
				
				for(int i=0;i<rand_itrs_ins;i++) {
					if(itrs_flag==0) {
						System.out.println(k+" - iteration: "+i);
						tim+=RankLib_x_ensemble(ntwk, num, num_links, i);
					}
					else {
						System.out.println(k+" - iteration: "+(rand_itrs+itrs_flag-1));
						tim+=RankLib_x_ensemble(ntwk, num, num_links, (rand_itrs+itrs_flag-1));
					}
					
					LinkedHashMap<String, LinkedHashMap<Integer, Double>> rank_scores_by_list=new LinkedHashMap<String, LinkedHashMap<Integer, Double>>();
					BufferedReader br=new BufferedReader(new FileReader("rank_score"));
					String line="";
					int cnt=0;
				    while ((line=br.readLine()) != null){
				    	String[] lines=line.split("\t");
				    	String nList=lines[0];
				    	double val=Double.parseDouble(lines[2]);
				    	
				    	if(rank_scores_by_list.containsKey(nList)) {
				    		rank_scores_by_list.get(nList).put(cnt, val);
				    	}
				    	else {
				    		LinkedHashMap<Integer, Double> tmp=new LinkedHashMap<Integer, Double>();
				    		tmp.put(cnt, val);
				    		rank_scores_by_list.put(nList, tmp);
				    	}
				    	
				    	if(!index_hit_cnt.containsKey(cnt) && itrs_flag==0) {
				    		index_hit_cnt.put(cnt, 0);
				    	}
				    	
				    	cnt++;
				    }   
					br.close();
					
					for(Map.Entry<String, LinkedHashMap<Integer, Double>> entry: rank_scores_by_list.entrySet()) {
						LinkedHashMap<Integer, Double> map=entry.getValue();
						HashMap<Integer, Double> map_sorted = sortByValue_Double(map);
						
						int cnt2=0;
						for(Map.Entry<Integer, Double> entry2: map_sorted.entrySet()) {
							if(cnt2>=Math.round((Quantile_count_test_low/(Quantile_count_test_low+Quantile_count_test_high))*map_sorted.size())) {
								index_hit_cnt.replace(entry2.getKey(), index_hit_cnt.get(entry2.getKey())+1);
							}
							cnt2++;
						}
					}
					
					
				}
				
				(new File("rank_score")).delete();
				
				index_hit_cnt_sorted=sortByValue_Integer(index_hit_cnt);
									
				index_hit_cnt_split_value=-1;
				int ahead_index_hit_cnt_split_value=-1;
				
				int cnt2=0;
				flag=false;
				for(Map.Entry<Integer, Integer> entry2: index_hit_cnt_sorted.entrySet()) {
					if(cnt2==Math.round((Quantile_count_test_low/(Quantile_count_test_low+Quantile_count_test_high))*index_hit_cnt_sorted.size())) {
						index_hit_cnt_split_value=entry2.getValue()>1 ? entry2.getValue() : 1;
						if(ahead_index_hit_cnt_split_value==index_hit_cnt_split_value) {
							System.out.println("There is a tie!");
							flag=true;
						}
						break;
						
					}
					else {
						ahead_index_hit_cnt_split_value=entry2.getValue();
					}
					cnt2++;
				}
				itrs_flag++;
			}
			
			for(Map.Entry<Integer, Integer> entry2: index_hit_cnt_sorted.entrySet()) {
				if(entry2.getValue()>=index_hit_cnt_split_value) {
					testlabel_1[entry2.getKey()]++;
				}
				else {
					testlabel_0[entry2.getKey()]++;
				}
			}
		}
		
		PrintWriter pw=new PrintWriter(new FileOutputStream(new File("time_ranking"), true)); 
		pw.println(ntwk+"\t"+tim);
		pw.close();
		
		int[] ranking_result=new int[testlabel_svm_ensemble.length];
		for(int i=0;i<testlabel_0.length;i++) {
			if(testlabel_0[i]>=testlabel_1[i]) {
				ranking_result[i]=0;
			}
			else {
				ranking_result[i]=1;
			}
		}
				
		F1Score_ranking(ranking_result, testlabel_svm_ensemble, prefix);
	}
	
    public static HashMap<Integer, Integer> sortByValue_Integer(HashMap<Integer, Integer> hm) { 
        List<Map.Entry<Integer, Integer> > list = new LinkedList<Map.Entry<Integer, Integer> >(hm.entrySet()); 
  
        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer> >() { 
            public int compare(Map.Entry<Integer, Integer> o1,  
                               Map.Entry<Integer, Integer> o2) 
            { 
                return (o1.getValue()).compareTo(o2.getValue()); 
            } 
        }); 
          
        HashMap<Integer, Integer> temp = new LinkedHashMap<Integer, Integer>(); 
        for (Map.Entry<Integer, Integer> aa : list) { 
            temp.put(aa.getKey(), aa.getValue()); 
        } 
        return temp; 
    } 
	
    public static HashMap<Integer, Double> sortByValue_Double(HashMap<Integer, Double> hm) { 
        List<Map.Entry<Integer, Double> > list = new LinkedList<Map.Entry<Integer, Double> >(hm.entrySet()); 
  
        Collections.sort(list, new Comparator<Map.Entry<Integer, Double> >() { 
            public int compare(Map.Entry<Integer, Double> o1,  
                               Map.Entry<Integer, Double> o2) 
            { 
                return (o1.getValue()).compareTo(o2.getValue()); 
            } 
        }); 
          
        HashMap<Integer, Double> temp = new LinkedHashMap<Integer, Double>(); 
        for (Map.Entry<Integer, Double> aa : list) { 
            temp.put(aa.getKey(), aa.getValue()); 
        } 
        return temp; 
    } 
    
    public static long RankLib_x_ensemble(String ntwk, int num, int num_links, int nitr) throws NumberFormatException, IOException, InterruptedException {
		long t=0;
    	if(nitr<1) {
			RankLib_tr_ensemble(ntwk, num, num_links);
			
			long t1=System.currentTimeMillis();
			RankLib_te_ensemble(ntwk, num, num_links);
			long t2=System.currentTimeMillis();
			t=t2-t1;
		}
		else {
			long t1=System.currentTimeMillis();
			RankLib_te_ensemble(ntwk, num, num_links);
			long t2=System.currentTimeMillis();
			t=t2-t1;
		}
    	return t;
	}
	
	public static void RankLib_tr_ensemble(String ntwk, int num, int num_links) throws IOException, InterruptedException {
		ArrayList<Integer> traintarget_rank_qids=new ArrayList<Integer>();
		int cqid=0;
		int cnt=0;
		for(int i=0;i<traintarget_rank_ensemble.length;i++) {
			if(cnt%ListSize==0){cqid++;}
			cnt++;
			traintarget_rank_qids.add(cqid);
		}
		data_index_shuffle(trainData_ensemble.rows, trainlabel_svm_ensemble);
		
		PrintWriter pw_tr=new PrintWriter(rand_dir+ntwk+"_tr_"+num+"_"+num_links);
		PrintWriter pw_tr_val=new PrintWriter(rand_dir+ntwk+"_tr_val_"+num+"_"+num_links);
		
		for(int i=0;i<traintarget_rank_ensemble.length;i++){
			pw_tr.print(traintarget_rank_ensemble[ORG2RDM.get(i)]+" qid:"+traintarget_rank_qids.get(i));
			pw_tr_val.println(traintarget_rank_ensemble[ORG2RDM.get(i)]);
			for(int j=0;j<trainData_ensemble.columns;j++){
				pw_tr.print(" "+(j+1)+":"+trainData_ensemble.get(ORG2RDM.get(i),j));
			}
			pw_tr.println();
		}
		pw_tr.close();
		pw_tr_val.close();
		
		System.out.println("java -jar "+rand_dir+"RankLib.jar"
				+ " -train "+rand_dir+ntwk+"_tr_"+num+"_"+num_links
				+ " -ranker 8 -metric2t NDCG@"+ListSize+" -gmax "+(int)(ListSize-1)+" -save "+rand_dir+ntwk+"_random_forests_model_"+num+"_"+num_links+"_"+ListSize+"_"+(int)(ListSize-1)
				);
			
/*		Runtime rt = Runtime.getRuntime();
		Process pr = rt.exec("java -jar "+rand_dir+"RankLib.jar"
					+ " -train "+rand_dir+ntwk+"_tr_"+num+"_"+num_links
					+ " -ranker 8 -metric2t NDCG@"+ListSize+" -gmax "+(int)(ListSize-1)+" -save "+rand_dir+ntwk+"_random_forests_model_"+num+"_"+num_links+"_"+ListSize+"_"+(int)(ListSize-1)
					+" -bag 1000 -shrinkage 0.01 -tc -1");
*/
		
		Runtime rt = Runtime.getRuntime();
		Process pr = rt.exec("java -jar "+rand_dir+"RankLib.jar"
					+ " -train "+rand_dir+ntwk+"_tr_"+num+"_"+num_links
					+ " -ranker 8 -metric2t NDCG@"+ListSize+" -gmax "+(int)(ListSize-1)+" -save "+rand_dir+ntwk+"_random_forests_model_"+num+"_"+num_links+"_"+ListSize+"_"+(int)(ListSize-1)
					);

		pr.waitFor();
		pr.destroy();
	}
	
	public static void data_index_shuffle(int size, int[] labels) {
		ORG2RDM=new HashMap<Integer, Integer>();
		ArrayList<Integer> idx_1=new ArrayList<Integer>();
		ArrayList<Integer> idx_0=new ArrayList<Integer>();
		for(int i=0;i<size;i++) {
			if(labels[i]==1) {
				idx_1.add(i);
			}
			else {
				idx_0.add(i);
			}
			
			
		}
		Collections.shuffle(idx_1);
		Collections.shuffle(idx_0);
		int cnt_1=0;
		int cnt_0=0;
		for(int i=0;i<size;i++) {
			if(i%2==0) {
				ORG2RDM.put(i, idx_1.get(cnt_1));
				cnt_1++;
			}
			else { 
				ORG2RDM.put(i, idx_0.get(cnt_0));
				cnt_0++;
			}
		}
	}
	
	public static void data_index_shuffle(int size) {
		ORG2RDM=new HashMap<Integer, Integer>();
		
		ArrayList<Integer> idx=new ArrayList<Integer>();
		for(int i=0;i<size;i++) {
			idx.add(i);
		}
		Collections.shuffle(idx);
		
		for(int i=0;i<size;i++) {
			ORG2RDM.put(i, idx.get(i));
		}
	}
	
	public static void RankLib_te_ensemble(String ntwk, int num, int num_links) throws IOException, InterruptedException {
		ArrayList<Integer> testtarget_rank_qids=new ArrayList<Integer>();
		int cqid=0;
		int cnt=0;
		for(int i=0;i<testData_ensemble.length;i++) {
			if(cnt%ListSize==0){cqid++;}
			cnt++;
			testtarget_rank_qids.add(cqid);
		}
//		data_index_shuffle(testData_ensemble.rows, testlabel_svm_ensemble);
		data_index_shuffle(testData_ensemble.rows);

		PrintWriter pw_te=new PrintWriter(rand_dir+ntwk+"_te_"+num+"_"+num_links);
		PrintWriter pw_te_val=new PrintWriter(rand_dir+ntwk+"_te_val_"+num+"_"+num_links);
		
		for(int i=0;i<testtarget_rank_ensemble.length;i++){
			pw_te.print(testtarget_rank_ensemble[ORG2RDM.get(i)]+" qid:"+testtarget_rank_qids.get(i));
			pw_te_val.println(testtarget_rank_ensemble[ORG2RDM.get(i)]);
			for(int j=0;j<testData_ensemble.columns;j++){
				pw_te.print(" "+(j+1)+":"+testData_ensemble.get(ORG2RDM.get(i),j));
			}
			pw_te.println();
		}
		pw_te.close();
		pw_te_val.close();
		
//		System.out.println("java -jar "+rand_dir+"RankLib.jar -load "+rand_dir+ntwk+"_random_forests_model_"+num+"_"+num_links+"_"+ListSize+"_"+(int)(ListSize-1)
//					+ " -rank "+rand_dir+ntwk+"_te_"+num+"_"+num_links+" -score rank_score_temporary");
		
		Runtime rt2 = Runtime.getRuntime();
 		Process pr2 = rt2.exec("java -jar "+rand_dir+"RankLib.jar -load "+rand_dir+ntwk+"_random_forests_model_"+num+"_"+num_links+"_"+ListSize+"_"+(int)(ListSize-1)
 					+ " -rank "+rand_dir+ntwk+"_te_"+num+"_"+num_links+" -score rank_score_temporary");
 		pr2.waitFor();
 		pr2.destroy();
 		
 		HashMap<Integer, String> rank_scores=new HashMap<Integer, String>();
 		BufferedReader br = new BufferedReader(new FileReader("rank_score_temporary"));
		String line="";
		cnt=0;
		while ((line=br.readLine()) != null) {
			rank_scores.put(ORG2RDM.get(cnt), line);
			cnt++;
		}
		br.close();
		
		PrintWriter pw=new PrintWriter("rank_score");
		for(int i=0;i<testData_ensemble.rows;i++) {
			pw.println(rank_scores.get(i));
		}
		
		pw.close();
		
		(new File("rank_score_temporary")).delete();
	}
	
	public static void rankScaling() throws NumberFormatException, IOException{
		int high_label_h=ListSize-1;
		int high_label_l=(int)((1-Quantile)*ListSize);
		
		int low_label_h=(int)((1-Quantile)*ListSize)-1;
		int low_label_l=0;
		
		double high_label_max=Double.MIN_VALUE;
		double high_label_min=Double.MAX_VALUE;
		
		double low_label_max=Double.MIN_VALUE;
		double low_label_min=Double.MAX_VALUE;
				
		for(int i=0;i<traintarget_regression.length;i++) {
			if(traintarget_regression[i]<Quantile_value) {
				if(traintarget_regression[i]>low_label_max) {
					low_label_max=traintarget_regression[i];
				}
				if(traintarget_regression[i]<low_label_min) {
					low_label_min=traintarget_regression[i];
				}
			}
			else {
				if(traintarget_regression[i]>high_label_max) {
					high_label_max=traintarget_regression[i];
				}
				if(traintarget_regression[i]<high_label_min) {
					high_label_min=traintarget_regression[i];
				}
			}
		}
		
		for(int i=0;i<testtarget_regression.length;i++) {
			if(testtarget_regression[i]<Quantile_value) {
				if(testtarget_regression[i]>low_label_max) {
					low_label_max=testtarget_regression[i];
				}
				if(testtarget_regression[i]<low_label_min) {
					low_label_min=testtarget_regression[i];
				}
			}
			else {
				if(testtarget_regression[i]>high_label_max) {
					high_label_max=testtarget_regression[i];
				}
				if(testtarget_regression[i]<high_label_min) {
					high_label_min=testtarget_regression[i];
				}
			}
		}
		
		traintarget_rank_ensemble=new int[traintarget_regression_ensemble.length];
		
		for(int i=0;i<traintarget_rank_ensemble.length;i++) {
			if(traintarget_regression_ensemble[i]<Quantile_value) {
				traintarget_rank_ensemble[i]=(int)((traintarget_regression_ensemble[i]-low_label_min)/(low_label_max-low_label_min)*(low_label_h-low_label_l)+low_label_l);
			}
			else {
				traintarget_rank_ensemble[i]=(int)((traintarget_regression_ensemble[i]-high_label_min)/(high_label_max-high_label_min)*(high_label_h-high_label_l)+high_label_l);
			}
		}
		
		testtarget_rank_ensemble=new int[testtarget_regression_ensemble.length];
		
		for(int i=0;i<testtarget_rank_ensemble.length;i++) {
			if(testtarget_regression_ensemble[i]<Quantile_value) {
				testtarget_rank_ensemble[i]=(int)((testtarget_regression_ensemble[i]-low_label_min)/(low_label_max-low_label_min)*(low_label_h-low_label_l)+low_label_l);
			}
			else {
				testtarget_rank_ensemble[i]=(int)((testtarget_regression_ensemble[i]-high_label_min)/(high_label_max-high_label_min)*(high_label_h-high_label_l)+high_label_l);
			}
		}
		
		
		
		traintarget_rank=new int[traintarget_regression.length];
		testtarget_rank=new int[testtarget_regression.length];
		
		for(int i=0;i<traintarget_rank.length;i++) {
			if(traintarget_regression[i]<Quantile_value) {
				traintarget_rank[i]=(int)((traintarget_regression[i]-low_label_min)/(low_label_max-low_label_min)*(low_label_h-low_label_l)+low_label_l);
			}
			else {
				traintarget_rank[i]=(int)((traintarget_regression[i]-high_label_min)/(high_label_max-high_label_min)*(high_label_h-high_label_l)+high_label_l);
			}
		}
		
		for(int i=0;i<testtarget_rank.length;i++) {
			if(testtarget_regression[i]<Quantile_value) {
				testtarget_rank[i]=(int)((testtarget_regression[i]-low_label_min)/(low_label_max-low_label_min)*(low_label_h-low_label_l)+low_label_l);
			}
			else {
				testtarget_rank[i]=(int)((testtarget_regression[i]-high_label_min)/(high_label_max-high_label_min)*(high_label_h-high_label_l)+high_label_l);
			}
		}
	}
	
	public static void dataShuffling(String ntwk, int num, int num_links) throws IOException {
		
		String x_feats="";
		switch (num) { 
        case 1: 
        	x_feats = ntwk+"_x_db_1_"+num_links; 
            break; 
        case 2: 
        	x_feats = ntwk+"_x_db_rb_2_"+num_links; 
            break; 
        case 3: 
        	x_feats = ntwk+"_x_db_dt_3_"+num_links; 
            break; 
        case 4: 
        	x_feats = ntwk+"_x_db_rt_4_"+num_links; 
            break; 
        case 5: 
        	x_feats = ntwk+"_x_db_rb_dt_5_"+num_links; 
            break; 
        case 6: 
        	x_feats = ntwk+"_x_db_rb_rt_6_"+num_links; 
            break; 
        case 7: 
        	x_feats = ntwk+"_x_db_dt_rt_7_"+num_links; 
            break; 
        case 8: 
        	x_feats = ntwk+"_x_db_rb_dt_rt_8_"+num_links; 
            break; 
        case 9: 
        	x_feats = ntwk+"_x_rb_9_"+num_links; 
            break; 
        case 10: 
        	x_feats = ntwk+"_x_rb_dt_10_"+num_links; 
            break; 
        case 11: 
        	x_feats = ntwk+"_x_rb_rt_11_"+num_links; 
            break; 
        case 12: 
        	x_feats = ntwk+"_x_rb_dt_rt_12_"+num_links; 
            break; 
        case 13: 
        	x_feats = ntwk+"_x_dt_13_"+num_links; 
            break; 
        case 14: 
        	x_feats = ntwk+"_x_dt_rt_14_"+num_links; 
            break; 
        case 15: 
        	x_feats = ntwk+"_x_rt_15_"+num_links; 
            break; 
        } 
		
		ArrayList<String> data=new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(x_feats));
		String line="";
		while ((line=br.readLine()) != null) {
			data.add(line);
		}
		br.close();
		
		ArrayList<Double> vals = new ArrayList<Double>();		
		br = new BufferedReader(new FileReader(ntwk+"_y_Scale_LOG_-2.0_"+num_links));
		while ((line=br.readLine()) != null){
	    	String[] lines=line.split("\t");
	    	double val=Double.parseDouble(lines[1]);
	    	vals.add(val);
	    }   
		br.close();		
		ArrayList<Double> vals_org = new ArrayList<Double>(vals);
		
		Collections.sort(vals, Collections.reverseOrder());
		int indx_split=(int)(Quantile*vals.size());
		Quantile_value=vals.get(indx_split-1);
		
		ArrayList<Integer> vals_high_indx = new ArrayList<Integer>();
		ArrayList<Integer> vals_low_indx = new ArrayList<Integer>();
		
		for(int i=0;i<vals_org.size();i++) {
			if(vals_org.get(i)<Quantile_value) {
				vals_low_indx.add(i);
			}
			else {
				vals_high_indx.add(i);
			}
		}
				
		Collections.shuffle(vals_high_indx);
		Collections.shuffle(vals_low_indx); 
				
		ArrayList<Integer> vals_high_indx_tr = new ArrayList<Integer>();
		ArrayList<Integer> vals_high_indx_te = new ArrayList<Integer>();
		ArrayList<Integer> vals_low_indx_tr = new ArrayList<Integer>();
		ArrayList<Integer> vals_low_indx_te = new ArrayList<Integer>();
		
		for(int i=0;i<vals_high_indx.size();i++) {
			if(i<Math.round(trainDataRatio*vals_high_indx.size())) {
				vals_high_indx_tr.add(vals_high_indx.get(i));
			}
			else {
				vals_high_indx_te.add(vals_high_indx.get(i));
			}
		}

		for(int i=0;i<vals_low_indx.size();i++) {
			if(i<Math.round(trainDataRatio*vals_low_indx.size())) {
				vals_low_indx_tr.add(vals_low_indx.get(i));
			}
			else {
				vals_low_indx_te.add(vals_low_indx.get(i));
			}
		}
		
		trainlabel_svm=new int[vals_high_indx_tr.size()+vals_low_indx_tr.size()];
		testlabel_svm=new int[vals_high_indx_te.size()+vals_low_indx_te.size()];
		traintarget_regression=new double[vals_high_indx_tr.size()+vals_low_indx_tr.size()];
		testtarget_regression=new double[vals_high_indx_te.size()+vals_low_indx_te.size()];
		
		trainData=new DoubleMatrix(vals_high_indx_tr.size()+vals_low_indx_tr.size(), data.get(0).split("\t").length-1);
		testData=new DoubleMatrix(vals_high_indx_te.size()+vals_low_indx_te.size(), data.get(0).split("\t").length-1);
		
		int cnt=0;
		for(int i:vals_high_indx_tr) {
			trainlabel_svm[cnt]=1;
			traintarget_regression[cnt]=vals_org.get(i);
			
			String[] data_s=data.get(i).split("\t");
			for(int j=1;j<data_s.length;j++) {
				trainData.put(cnt, j-1, Double.parseDouble(data_s[j]));
			}
			cnt++;
		}
		
		for(int i:vals_low_indx_tr) {
			trainlabel_svm[cnt]=0;
			traintarget_regression[cnt]=vals_org.get(i);
			
			String[] data_s=data.get(i).split("\t");
			for(int j=1;j<data_s.length;j++) {
				trainData.put(cnt, j-1, Double.parseDouble(data_s[j]));
			}
			cnt++;
		}

//		PrintWriter pw_2c=new PrintWriter(ntwk+"_test_real_2c");
//		PrintWriter pw_val=new PrintWriter(ntwk+"_test_real_val");
		cnt=0;
		for(int i:vals_high_indx_te) {
			testlabel_svm[cnt]=1;
//			pw_2c.println(testlabel_svm[cnt]);
			testtarget_regression[cnt]=vals_org.get(i);
//			pw_val.println(testtarget_regression[cnt]);
			
			String[] data_s=data.get(i).split("\t");
			for(int j=1;j<data_s.length;j++) {
				testData.put(cnt, j-1, Double.parseDouble(data_s[j]));
			}
			cnt++;
		}
		for(int i:vals_low_indx_te) {
			testlabel_svm[cnt]=0;
//			pw_2c.println(testlabel_svm[cnt]);
			testtarget_regression[cnt]=vals_org.get(i);
//			pw_val.println(testtarget_regression[cnt]);
			
			String[] data_s=data.get(i).split("\t");
			for(int j=1;j<data_s.length;j++) {
				testData.put(cnt, j-1, Double.parseDouble(data_s[j]));
			}
			cnt++;
		}
//		pw_2c.close();
//		pw_val.close();

		vals_high_indx_te_cnt=vals_high_indx_te.size();
	}
	
	public static long run_svm(DoubleMatrix trData, DoubleMatrix teData, int[] trLabel, int[] teLabel, String results) throws IOException{
		PrintWriter pw_tr=new PrintWriter("tr_data");
		for(int i=0;i<trLabel.length;i++){
			pw_tr.print(trLabel[i]);
			for(int j=0;j<trData.columns;j++)
				pw_tr.print(" "+(j+1)+":"+trData.get(i, j));
			pw_tr.println();
		}
		pw_tr.close();
		
		PrintWriter pw_te=new PrintWriter("te_data");
		for(int i=0;i<teLabel.length;i++){
			pw_te.print(teLabel[i]);
			for(int j=0;j<teData.columns;j++)
				pw_te.print(" "+(j+1)+":"+teData.get(i, j));
			pw_te.println();
		}
		pw_te.close();
		
		(new File("Cross_Validation_Results.txt")).delete();
/*		ExecutorService executor = Executors.newFixedThreadPool(SizeOfThreadPool);
		for(double g_ml:param_g_ml){
			for(double c:param_c){
				Cross_Validation_Runnable cv_run=new Cross_Validation_Runnable(c+"", g_ml+"", folder_cv);
				executor.execute(cv_run);
			}
		}
		executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all cross-validation threads!");
		
		BufferedReader br = new BufferedReader(new FileReader("Cross_Validation_Results.txt"));
		double best_c=1.0;
		double best_g_ml=1.0;
		double accuracy=-1;
		double nSV=Double.POSITIVE_INFINITY;
		String line="";
		while ((line=br.readLine()) != null) {
			String[] data=line.split("\t");
			
			if(Double.parseDouble(data[3])>thres_cv) { // accuracy > threshold
				if(Double.parseDouble(data[2])<nSV) {
					accuracy=Double.parseDouble(data[3]);
					best_c=Double.parseDouble(data[0]);
					best_g_ml=Double.parseDouble(data[1]);
					nSV=Double.parseDouble(data[2]);
				}
				else if(Double.parseDouble(data[2])==nSV && Double.parseDouble(data[3])>accuracy) {
					accuracy=Double.parseDouble(data[3]);
					best_c=Double.parseDouble(data[0]);
					best_g_ml=Double.parseDouble(data[1]);
					nSV=Double.parseDouble(data[2]);
				}
			}
			else {
				if(Double.parseDouble(data[3])>accuracy) {
					accuracy=Double.parseDouble(data[3]);
					best_c=Double.parseDouble(data[0]);
					best_g_ml=Double.parseDouble(data[1]);
					nSV=Double.parseDouble(data[2]);
				}
				else if(Double.parseDouble(data[3])==accuracy && Double.parseDouble(data[2])<nSV) {
					accuracy=Double.parseDouble(data[3]);
					best_c=Double.parseDouble(data[0]);
					best_g_ml=Double.parseDouble(data[1]);
					nSV=Double.parseDouble(data[2]);
				}
			}
		}
		br.close();
		
		*/
		
		(new File("tr_model")).delete();
		
		String[] trainArgs = {"-g", Math.pow(2, 0)+"", "-c", Math.pow(2, 0)+"", //"-h", "0", 
				"tr_data", "tr_model"};
		svm_train.main(trainArgs, "");
		
		String[] testArgs = {"te_data", "tr_model", results};
		
		long t1=System.currentTimeMillis();
		svm_predict.main(testArgs);
		long t2=System.currentTimeMillis();
		
		(new File("tr_data")).delete();
		(new File("te_data")).delete();
		(new File("tr_model")).delete();
		
//		double[] retn=new double[2];
//		retn[0]=best_g_ml;
//		retn[1]=best_c;
		return (t2-t1);
		
	}
	
	public static double F1Score_regression(int[] target, int[] real, String prefix) throws NumberFormatException, IOException{
		HashMap<Integer, ArrayList<Integer>> label_idxs=new HashMap<Integer, ArrayList<Integer>>();
		for(int i=0;i<testlabel_svm.length;i++) {
			if(label_idxs.containsKey(testlabel_svm[i])) {
				label_idxs.get(testlabel_svm[i]).add(i);
			}
			else {
				ArrayList<Integer> tmp=new ArrayList<Integer>();
				tmp.add(i);
				label_idxs.put(testlabel_svm[i], tmp);
			}
		}
		
		ArrayList<Integer> label_0=new ArrayList<Integer>(label_idxs.get(0));
		ArrayList<Integer> label_1=new ArrayList<Integer>(label_idxs.get(1));
				
		Collections.shuffle(label_0);
		
		ArrayList<Integer> test_data_considered=new ArrayList<Integer>();
		
		for(int i=0;i<label_1.size();i++) {
			test_data_considered.add(label_1.get(i));
			test_data_considered.add(label_0.get(i));
		}		
		
		HashMap<Integer, Integer> labelIndexMap=new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> labelIndexMap_inv=new HashMap<Integer, Integer>();
		
		HashSet<Integer> labelSet=new HashSet<Integer>();
		int ct=0;
		for(int i=0;i<real.length;i++){
			labelSet.add(real[i]);
			if(!labelIndexMap.containsKey(real[i])){
				labelIndexMap.put(real[i], ct);
				labelIndexMap_inv.put(ct, real[i]);
				ct++;
			}
		}		
		
		int[] labelSize=new int[labelSet.size()];
		for(int i=0;i<labelSize.length;i++){
			labelSize[i]=0;
		}
		
		for(int i=0;i<real.length;i++){
			if(test_data_considered.contains(i))
				labelSize[labelIndexMap.get(real[i])]++;
		}
		
		int[][] ConfusionMatrix=new int[labelSet.size()][labelSet.size()];
		for(int i=0;i<labelSet.size();i++){
			for(int j=0;j<labelSet.size();j++){
				ConfusionMatrix[i][j]=0;
			}
		}
		
		for(int i=0;i<real.length;i++){
			if(test_data_considered.contains(i))
				ConfusionMatrix[labelIndexMap.get(target[i])][labelIndexMap.get(real[i])]++;
		}
		
		double total_F1=0;
		
		for(int i=0;i<labelSet.size();i++){
			int all_p=0;
			int real_p=0;
			for(int j=0;j<labelSet.size();j++){
				all_p=all_p+ConfusionMatrix[i][j];
				real_p=real_p+ConfusionMatrix[j][i];
			}
			
			double precision=(double)ConfusionMatrix[i][i]/(double)all_p;
			double recall=(double)ConfusionMatrix[i][i]/(double)real_p;
			
			total_F1=total_F1+2*(precision*recall)/(precision+recall)*(double)(labelSize[i])/(double)test_data_considered.size();
						
			prefix=prefix+"\t"+(int)(double)labelIndexMap_inv.get(i)+","+precision+","+recall+","+labelSize[i];
		}
		prefix=prefix+"\t"+total_F1;
		
		String timeStamp = new SimpleDateFormat("yyyy/MM/dd-HH:mm:ss").format(new Date());
		PrintWriter pwx=new PrintWriter(new FileOutputStream("regression_ensemble_results.txt", true));
		pwx.println(timeStamp+"\t"+prefix);
		pwx.close();
		
		return total_F1;
	}
	
	public static double F1Score_ranking(int[] target, int[] real, String prefix) throws NumberFormatException, IOException{
		HashMap<Integer, Integer> labelIndexMap=new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> labelIndexMap_inv=new HashMap<Integer, Integer>();
		
		HashSet<Integer> labelSet=new HashSet<Integer>();
		int ct=0;
		for(int i=0;i<real.length;i++){
			labelSet.add(real[i]);
			if(!labelIndexMap.containsKey(real[i])){
				labelIndexMap.put(real[i], ct);
				labelIndexMap_inv.put(ct, real[i]);
				ct++;
			}
		}
				
		int[] labelSize=new int[labelSet.size()];
		for(int i=0;i<labelSize.length;i++){
			labelSize[i]=0;
		}
		
		for(int i=0;i<real.length;i++){
				labelSize[labelIndexMap.get(real[i])]++;
		}
		
		int[][] ConfusionMatrix=new int[labelSet.size()][labelSet.size()];
		for(int i=0;i<labelSet.size();i++){
			for(int j=0;j<labelSet.size();j++){
				ConfusionMatrix[i][j]=0;
			}
		}
		
		for(int i=0;i<real.length;i++){
				ConfusionMatrix[labelIndexMap.get(target[i])][labelIndexMap.get(real[i])]++;
		}
		
		double total_F1=0;
		
		for(int i=0;i<labelSet.size();i++){
			
			int all_p=0;
			int real_p=0;
			for(int j=0;j<labelSet.size();j++){
				all_p=all_p+ConfusionMatrix[i][j];
				real_p=real_p+ConfusionMatrix[j][i];
			}
			
			double precision=(double)ConfusionMatrix[i][i]/(double)all_p;
			double recall=(double)ConfusionMatrix[i][i]/(double)real_p;
			
			total_F1=total_F1+2*(precision*recall)/(precision+recall)*(double)(labelSize[i])/(double)real.length;
						
			prefix=prefix+"\t"+(int)(double)labelIndexMap_inv.get(i)+","+precision+","+recall+","+labelSize[i];
		}
		prefix=prefix+"\t"+total_F1;
		
		String timeStamp = new SimpleDateFormat("yyyy/MM/dd-HH:mm:ss").format(new Date());
		PrintWriter pwx=new PrintWriter(new FileOutputStream("ranking_ensemble_results.txt", true));
		pwx.println(timeStamp+"\t"+prefix);
		pwx.close();
		
		return total_F1;
	}
	
	public static String F1Score_classification(String resultFile) throws NumberFormatException, IOException{
		HashMap<Integer, ArrayList<Integer>> label_idxs=new HashMap<Integer, ArrayList<Integer>>();
		for(int i=0;i<testlabel_svm.length;i++) {
			if(label_idxs.containsKey(testlabel_svm[i])) {
				label_idxs.get(testlabel_svm[i]).add(i);
			}
			else {
				ArrayList<Integer> tmp=new ArrayList<Integer>();
				tmp.add(i);
				label_idxs.put(testlabel_svm[i], tmp);
			}
		}
		
		ArrayList<Integer> label_0=new ArrayList<Integer>(label_idxs.get(0));
		ArrayList<Integer> label_1=new ArrayList<Integer>(label_idxs.get(1));
		
		Collections.shuffle(label_0);
		
		ArrayList<Integer> test_data_considered=new ArrayList<Integer>();
		
		for(int i=0;i<label_1.size();i++) {
			test_data_considered.add(label_1.get(i));
			test_data_considered.add(label_0.get(i));
		}
		
		HashMap<Integer, Integer> labelIndexMap_inv=new HashMap<Integer, Integer>();
		
		HashMap<Integer, Integer> labelIndexMap=new HashMap<Integer, Integer>();
		HashSet<Integer> labelSet=new HashSet<Integer>();
		int ct=0;
		for(int i=0;i<testlabel_svm.length;i++){
			labelSet.add(testlabel_svm[i]);
			if(!labelIndexMap.containsKey(testlabel_svm[i])){
				labelIndexMap.put(testlabel_svm[i], ct);
				labelIndexMap_inv.put(ct, testlabel_svm[i]);
				ct++;
			}
		}
		
		int[] labelSize=new int[labelSet.size()];
		for(int i=0;i<labelSize.length;i++)
			labelSize[i]=0;
		
		for(int i=0;i<testlabel_svm.length;i++)
			if(test_data_considered.contains(i))
				labelSize[labelIndexMap.get(testlabel_svm[i])]++;
		
		BufferedReader br = new BufferedReader(new FileReader(resultFile));
		int[] testLabel=new int[testlabel_svm.length];
		String testLabelS="";
		ct=0;
		while ((testLabelS=br.readLine()) != null) {
			testLabel[ct]=(int)Double.parseDouble(testLabelS);
			ct++;
		}
		br.close();
		
		int[][] ConfusionMatrix=new int[labelSet.size()][labelSet.size()];
		for(int i=0;i<labelSet.size();i++)
			for(int j=0;j<labelSet.size();j++)
				ConfusionMatrix[i][j]=0;
		
		for(int i=0;i<testlabel_svm.length;i++)
			if(test_data_considered.contains(i))
				ConfusionMatrix[labelIndexMap.get(testLabel[i])][labelIndexMap.get(testlabel_svm[i])]++;
		
		String val="";
		double total_F1=0;
		for(int i=0;i<labelSet.size();i++){
			int all_p=0;
			int real_p=0;
			for(int j=0;j<labelSet.size();j++){
				all_p=all_p+ConfusionMatrix[i][j];
				real_p=real_p+ConfusionMatrix[j][i];
			}
			double precision=(double)ConfusionMatrix[i][i]/(double)all_p;
			double recall=(double)ConfusionMatrix[i][i]/(double)real_p;
			total_F1=total_F1+2*(precision*recall)/(precision+recall)*(double)(labelSize[i])/(double)test_data_considered.size();
			
			val=val+"\t"+labelIndexMap_inv.get(i)+","+precision+","+recall+","+labelSize[i];
		}
		val=val+"\t"+total_F1;
		return val;
	}

	/*
	 * long t1=System.currentTimeMillis(); dataGeneration(k, id); long
	 * t2=System.currentTimeMillis();
	 * 
	 * PrintWriter pw2 = new PrintWriter(new FileOutputStream(new File("sim_time"),
	 * true)); pw2.println(k+"\t"+id+"\t"+(t2-t1)); pw2.close();
	 */
	public static void dataGeneration(String ntwk, int id) throws NumberFormatException, IOException, InterruptedException{
		HashMap<String, String> Node2Link=new HashMap<String, String>();
		HashMap<String, String> Link2Node=new HashMap<String, String>();
		
		(new File(ntwk+"_val.txt")).delete();
		
		BufferedReader br=new BufferedReader(new FileReader("Link2Node/"+ntwk+"_Link2Node"));
		String line="";
		int nLinks=0;
		while ((line=br.readLine()) != null){
			String[] lines=line.split("\t");
			Node2Link.put(lines[1], lines[0]);
			Link2Node.put(lines[0], lines[1]);
			nLinks++;
		}
		br.close();
		
		int xlab=-1;
		int[] Links=new int[nLinks];
		int idx=0;
		br=new BufferedReader(new FileReader(ntwk+"/orig/"+ntwk+"_net.txt"));
		boolean flag=false;
		LinkedHashMap<String, String> data=new LinkedHashMap<String, String>();
	    while ((line=br.readLine()) != null){
	    	
	    	if(flag){
	    		String[] lines=line.split("\t");	
	    		int src_node=Integer.parseInt(lines[1].replaceAll("\\s+",""));
	    		int dst_node=Integer.parseInt(lines[2].replaceAll("\\s+",""));
	    		
	    		if(Link2Node.containsKey(src_node+","+dst_node)){
	    			data.put(Link2Node.get(src_node+","+dst_node), line);
	    			Links[idx]=Integer.parseInt(Link2Node.get(src_node+","+dst_node));
	    			idx++;
	    		}
	    		else{
	    			data.put(xlab+"", line);
	    			xlab--;
	    		}
	    		
	    	}
	    	
	    	if(line.contains("~")){
	    		flag=true;
	    	}
	    }   
		br.close();
		
		Set<String> nodes=Node2Link.keySet();
		
		int memory=Integer.MIN_VALUE;
		for(int r=max_links;r<=max_links;r++){
			if(memory<nLinksets && memory<CombinatoricsUtils.binomialCoefficient(nodes.size(), r)){
				memory=(int)CombinatoricsUtils.binomialCoefficient(nodes.size(), r);
			}
			
			int num=nLinksets<memory ? nLinksets : memory;
			ArrayList<HashSet<String>> cbs=new ArrayList<HashSet<String>>();
			ArrayList<HashSet<String>> cbs_good=new ArrayList<HashSet<String>>();
			int ccnt=0;
			while(cbs_good.size()<num && cbs.size()<Integer.MAX_VALUE && cbs.size()<CombinatoricsUtils.binomialCoefficient(nodes.size(), r)){
				long t1=System.currentTimeMillis();
				
				ArrayList<String> nodes_copy=new ArrayList<String>(nodes);
				HashSet<String> tmp=new HashSet<String>();
				Random rr = new Random();
				
				for(int i=0;i<r;i++){
					int idxx=rr.nextInt(nodes_copy.size());
					tmp.add(nodes_copy.get(idxx));
					nodes_copy.remove(idxx);
				}
				boolean flagg=true;
				for(HashSet<String> i:cbs){
					if(i.containsAll(tmp)){
						flagg=false;
						break;
					}
				}
				
				ccnt++;
				
				if(flagg){
					ArrayList<String> cbns_a=new ArrayList<String>(tmp);
					String lks=cbns_a.get(0);
					for(int k=1;k<tmp.size();k++){
						lks=lks+","+cbns_a.get(k);
					}
					cbs.add(tmp);
					
					if(validCombi(ntwk, lks, data, id)) {
						
						cbs_good.add(tmp);
						
						PrintWriter pw2 = new PrintWriter(new FileOutputStream(new File("sim_time_single"),true)); 
						long t2=System.currentTimeMillis();
						pw2.println(ntwk+"\t"+id+"\t"+cbs_good.size()+"\t"+(t2-t1)); 
						pw2.close();
					}
					
					System.out.println(num+"\t"+cbs_good.size()+" - "+cbs.size()+" - "+ccnt);
				}
			}
		}
	}
	
	public static boolean validCombi(String ntwk, String str,  LinkedHashMap<String, String> data, int id) throws NumberFormatException, IOException, InterruptedException {
		boolean res=false;
//		PrintWriter pw2 = new PrintWriter(new FileOutputStream(
//			    new File(ntwk+"_val.txt"), 
//			    true /* append = true */)); 
		
		String[] strs=str.split(",");
    	HashSet<String> strSet=new HashSet<String>(Arrays.asList(strs));
    	
    	PrintWriter pw=new PrintWriter(ntwk+"/"+ntwk+"_net.txt");
		
    	BufferedReader br=new BufferedReader(new FileReader(ntwk+"/orig/"+ntwk+"_net.txt"));
    	String line="";
	    while ((line=br.readLine()) != null){
	    	if(line.contains("<NUMBER OF LINKS>")){	    		
	    		pw.println("<NUMBER OF LINKS> "+(Integer.parseInt(line.split(">")[1].replaceAll("\\s+",""))-strSet.size()));
	    	}
	    	else if(line.contains("~")){
	    		pw.println(line);
	    		break;
	    	}
	    	else{
	    		pw.println(line);
	    	}
	    }   
		br.close();
		
		for(Map.Entry<String, String> entry: data.entrySet()){
			if(!strSet.contains(entry.getKey())){
				pw.println(entry.getValue());
			}
		}
		
		pw.close();
		
		
		HashSet<String> s1=new HashSet<String>();
		br=new BufferedReader(new FileReader(ntwk+"/"+ntwk+alg+".txt"));
	    while ((line=br.readLine()) != null){
	    	s1.add(line);
	    }   
		br.close();
		
		
		
		System.out.println(str);
		Runtime rt = Runtime.getRuntime();
		Process pr = rt.exec("/home/durham314/Downloads/TrafficAssign-master/assign /home/durham314/Downloads/TrafficAssign-master/s_"+ntwk+"_"+id+".txt");
		pr.waitFor();
		
		HashSet<String> s2=new HashSet<String>();
		br=new BufferedReader(new FileReader(ntwk+"/"+ntwk+alg+".txt"));
	    while ((line=br.readLine()) != null){
	    	s2.add(line);
	    }   
		br.close();
		
		if(!s1.equals(s2)){
			System.out.println("Good!");
			res=true;
			
			br=new BufferedReader(new FileReader(ntwk+"/"+ntwk+alg+".txt"));
		    while ((line=br.readLine()) != null){
		    	if(line.contains("<system cost>")){	    		
//		    		pw2.println(str+"\t"+(Double.parseDouble(line.split(">")[1].replaceAll("\\s+","")))+"");
		    		break;
		    	}
		    }   
			br.close();
		}
//		pw2.close();
		
		return res;
	}
}
