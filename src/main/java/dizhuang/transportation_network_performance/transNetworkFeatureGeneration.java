package dizhuang.transportation_network_performance;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.CombinatoricsUtils;

import smile.clustering.KMeans;
import smile.math.Math;

public class transNetworkFeatureGeneration {
	
	public static String[] algs= {"UE", "SUE", "SO"};
	public static String[] algms= {"_FW_0.02_0.10_2.00_flow", "_DSD_0.02_0.10_2.00_flow", "_COL_0.02_0.10_2.00_flow"};
	public static int alg_idx=0;
	private static double lownew=0.0;
	private static double highnew=1.0;
	public static int nLinksets=10000;
	public static int max_links=5;
	public static int num_links=4;
	
	public static LinkedHashSet<String> setLinkset=new LinkedHashSet<String>();
		
	public static void simulation(String ntwk) throws NumberFormatException, IOException, InterruptedException {
		System.out.println(ntwk+" data generation . . .");
//		bagOfWords(ntwk);
		alg_idx=0;
//		dataGeneration(ntwk);
//		alg_idx=2;
//		dataGeneration(ntwk);
//		alg_idx=2;
//		dataGeneration(ntwk);

//		chkLinkFeats(ntwk);
		
		chkLinkFeats_num_links_345(ntwk);
		dataGeneration_bow(ntwk);
		
		DateScale_y(ntwk, -2.0);
	}
	
	public static void main( String[] args ) throws IOException, InterruptedException{
/*		num_links=3;
		simulation("SiouxFalls"); // [-2.0,-2.4]
		simulation("Anaheim"); // [-2.0,-2.4]
		simulation("Mitte"); // [-2.0,-2.4]
		simulation("Prenzlauerberg"); // [-2.0,-2.4]
		simulation("Tiergarten"); // [-2.0,-2.4]
		simulation("Friedrichshain"); // [-2.0,-2.4]
		simulation("Berlin3c"); // [-2.0,-2.4]
		
		num_links=4;
		simulation("SiouxFalls");
		simulation("Anaheim");
		simulation("Mitte");
		simulation("Prenzlauerberg");
		simulation("Tiergarten");
		simulation("Friedrichshain");
		simulation("Berlin3c");
		
		num_links=5;
		simulation("SiouxFalls");
		simulation("Anaheim");
		simulation("Mitte");
		simulation("Prenzlauerberg");
		simulation("Tiergarten");
		simulation("Friedrichshain");
		simulation("Berlin3c");
*/		
		
//		num_links=345;
//		simulation("Berlin3c"); 
		
		num_links=345;
		simulation("Anaheim"); 
		simulation("Tiergarten"); 
		simulation("SiouxFalls"); 
		simulation("Berlin3c"); 
		
		
		

	}
	
	public static void DateScale_y(String ntwk, double lambda) throws IOException{
		String line="";
		BufferedReader br=new BufferedReader(new FileReader(ntwk+"_y_"+num_links));
		double high=Double.MIN_VALUE;
		double low=Double.MAX_VALUE;
		double high_log=Double.MIN_VALUE;
		double low_log=Double.MAX_VALUE;
		double high_log_lambda=Double.MIN_VALUE;
		double low_log_lambda=Double.MAX_VALUE;
		
		
	    while ((line=br.readLine()) != null){
	    	String[] lines=line.split("\t");
	    	if(high<Double.parseDouble(lines[1])){
	    		high=Double.parseDouble(lines[1]);
	    	}
	    	if(low>Double.parseDouble(lines[1])){
	    		low=Double.parseDouble(lines[1]);
	    	}
	    	
	    	if(high_log<Math.log(Double.parseDouble(lines[1]))){
	    		high_log=Math.log(Double.parseDouble(lines[1]));
	    	}
	    	if(low_log>Math.log(Double.parseDouble(lines[1]))){
	    		low_log=Math.log(Double.parseDouble(lines[1]));
	    	}
	    	
	    	if(high_log_lambda<((Math.pow(Double.parseDouble(lines[1]), lambda)-1.0)/lambda)){
	    		high_log_lambda=(Math.pow(Double.parseDouble(lines[1]), lambda)-1.0)/lambda;
	    	}
	    	if(low_log_lambda>((Math.pow(Double.parseDouble(lines[1]), lambda)-1.0)/lambda)){
	    		low_log_lambda=(Math.pow(Double.parseDouble(lines[1]), lambda)-1.0)/lambda;
	    	}
	    }   
		br.close();
		
		
		PrintWriter pw_y=new PrintWriter(ntwk+"_y_Scale_"+num_links);
		PrintWriter pw_y_log=new PrintWriter(ntwk+"_y_Scale_LOG_"+num_links);
		PrintWriter pw_y_log_lambda=new PrintWriter(ntwk+"_y_Scale_LOG_"+lambda+"_"+num_links);
		br=new BufferedReader(new FileReader(ntwk+"_y_"+num_links));
	    while ((line=br.readLine()) != null){
	    	String[] lines=line.split("\t");
	    	double dts=Double.parseDouble(lines[1]);
	    	double dts_log=Math.log(Double.parseDouble(lines[1]));
	    	double dts_log_lambda=(Math.pow(Double.parseDouble(lines[1]), lambda)-1.0)/lambda;
	    	
	    	dts=(dts-low)/(high-low)*(highnew-lownew)+lownew;
	    	
	    	dts_log=(dts_log-low_log)/(high_log-low_log)*(highnew-lownew)+lownew;
	    	
	    	dts_log_lambda=(dts_log_lambda-low_log_lambda)/(high_log_lambda-low_log_lambda)*(highnew-lownew)+lownew;
	    		    	
	    	pw_y.println(lines[0]+"\t"+dts);
	    	
	    	pw_y_log.println(lines[0]+"\t"+dts_log);
	    	
	    	pw_y_log_lambda.println(lines[0]+"\t"+dts_log_lambda);
	    }   
		br.close();
		pw_y.close();
		pw_y_log.close();
		pw_y_log_lambda.close();
	}
	
	public static void dataGeneration_bow(String ntwk) throws IOException{
		BufferedReader br=new BufferedReader(new FileReader("link_words/"+ntwk+"_x_4F2_s_Words"));
		String line=br.readLine();
		String[] ls=line.split("\t");
		
		ArrayList<String> totalNets=new ArrayList<String>();
		HashMap<String, String> words=new HashMap<String, String>();
		
		while ((line=br.readLine()) != null){
			String[] lines=line.split("\t");
			
			words.put(lines[0], line);
			
			totalNets.add(lines[0]);
		}
		br.close();
		
		br=new BufferedReader(new FileReader(ntwk+"_val_rd_"+num_links+".txt"));
		
		PrintWriter pw_y=new PrintWriter(ntwk+"_y_"+num_links);
		
		int cnt=0;
		
		int totalFeats=0;
		for(int i=1;i<ls.length;i++){
			totalFeats+=(int)Double.parseDouble(ls[i]);
		}
		totalFeats=totalFeats*2;
		
		int[] IDFs=new int[totalFeats];
		
		int total_nets=0;
		
		LinkedHashMap<Integer, ArrayList<Double>> all_data_feats_deleted_db=new LinkedHashMap<Integer, ArrayList<Double>>();
		LinkedHashMap<Integer, ArrayList<Double>> all_data_feats_residual_db=new LinkedHashMap<Integer, ArrayList<Double>>();
		
		while ((line=br.readLine()) != null){
			total_nets++;
			
			String[] lines=line.split("\t");
			String val=lines[1];
			HashSet<String> links_deleted=new HashSet<String>();
			if(lines[0].contains(",")){
				links_deleted.addAll(Arrays.asList(lines[0].split(",")));
			}
			else{
				links_deleted.add(lines[0]);
			}
			
			ArrayList<Double> Feats_deleted_db=new ArrayList<Double>();
			ArrayList<Double> Feats_residual_db=new ArrayList<Double>();
			int totalFeats_cnt=0;
			for(int i=1;i<ls.length;i++){
				int[] feats_deleted=new int[(int)Double.parseDouble(ls[i])];
				int[] feats_residual=new int[(int)Double.parseDouble(ls[i])];
				
				for(String str: links_deleted){
					String word=words.get(str);
					int vls=(int)Double.parseDouble(word.split("\t")[i]);
					feats_deleted[vls]++;
				}
				
				for(String str: totalNets){
					if(!links_deleted.contains(str)){
						String word=words.get(str);
						int vls=(int)Double.parseDouble(word.split("\t")[i]);
						feats_residual[vls]++;
					}
				}
				
				int sum=0;
				for(int k=0;k<feats_deleted.length;k++){
					sum+=feats_deleted[k];
				}
				double[] feats_deleted_db=new double[feats_deleted.length]; // double feature
				for(int k=0;k<feats_deleted.length;k++){
					feats_deleted_db[k]=(double)feats_deleted[k]/(double)sum;
				}
				
				sum=0;
				for(int k=0;k<feats_residual.length;k++){
					sum+=feats_residual[k];
				}
				double[] feats_residual_db=new double[feats_residual.length];
				for(int k=0;k<feats_residual.length;k++){
					feats_residual_db[k]=(double)feats_residual[k]/(double)sum;
				}
				
				
				for(double k: feats_deleted_db){
					Feats_deleted_db.add(k);
					if(k>0){
						IDFs[totalFeats_cnt]++;
					}
					totalFeats_cnt++;
				}
				
				for(double k: feats_residual_db){
					Feats_residual_db.add(k);
					if(k>0){
						IDFs[totalFeats_cnt]++;
					}
					totalFeats_cnt++;
				}
			}
			pw_y.println(cnt+"\t"+val);
			
			all_data_feats_deleted_db.put(cnt, Feats_deleted_db); // db
			all_data_feats_residual_db.put(cnt, Feats_residual_db); // rb
			
			cnt++;
		}
		br.close();
		pw_y.close();
		
		double[] IDFs_db=new double[IDFs.length];
		for(int i=0;i<IDFs.length;i++){
			IDFs_db[i]=(double)total_nets/(double)IDFs[i];
		}
		
		LinkedHashMap<Integer, String> x_db=new LinkedHashMap<Integer, String>();
		LinkedHashMap<Integer, String> x_dt=new LinkedHashMap<Integer, String>();
		LinkedHashMap<Integer, String> x_rb=new LinkedHashMap<Integer, String>();
		LinkedHashMap<Integer, String> x_rt=new LinkedHashMap<Integer, String>();
				
		for(Map.Entry<Integer, ArrayList<Double>> entry: all_data_feats_deleted_db.entrySet()){ // db dt
			String db="";
			for(double fts: entry.getValue()){
				db+="\t"+fts;
			}
			x_db.put(entry.getKey(), db);
			
			String dt="";
			int idx=0;
			for(double fts: entry.getValue()){
				dt+="\t"+fts*Math.log(IDFs_db[idx]);
				idx++;
			}
			x_dt.put(entry.getKey(), dt);
		}
		
		for(Map.Entry<Integer, ArrayList<Double>> entry: all_data_feats_residual_db.entrySet()){ // rb rt
			String rb="";
			for(double fts: entry.getValue()){
				rb+="\t"+fts;
			}
			x_rb.put(entry.getKey(), rb);
			
			String rt="";
			int idx=0;
			for(double fts: entry.getValue()){
				rt+="\t"+fts*Math.log(IDFs_db[idx]);
				idx++;
			}
			x_rt.put(entry.getKey(), rt);
		}
		
//		1 - pw_x_db
		PrintWriter pw_x_db=new PrintWriter(ntwk+"_x_db_1_"+num_links);
//		2 - pw_x_db_rb
		PrintWriter pw_x_db_rb=new PrintWriter(ntwk+"_x_db_rb_2_"+num_links);
//		3 - pw_x_db_dt
		PrintWriter pw_x_db_dt=new PrintWriter(ntwk+"_x_db_dt_3_"+num_links);
//		4 - pw_x_db_rt
		PrintWriter pw_x_db_rt=new PrintWriter(ntwk+"_x_db_rt_4_"+num_links);

//		5 - pw_x_db_rb_dt
		PrintWriter pw_x_db_rb_dt=new PrintWriter(ntwk+"_x_db_rb_dt_5_"+num_links);
//		6 - pw_x_db_rb_rt
		PrintWriter pw_x_db_rb_rt=new PrintWriter(ntwk+"_x_db_rb_rt_6_"+num_links);
//		7 - pw_x_db_dt_rt
		PrintWriter pw_x_db_dt_rt=new PrintWriter(ntwk+"_x_db_dt_rt_7_"+num_links);

//		8 - pw_x_db_rb_dt_rt
		PrintWriter pw_x_db_rb_dt_rt=new PrintWriter(ntwk+"_x_db_rb_dt_rt_8_"+num_links);

//		9 - pw_x_rb
		PrintWriter pw_x_rb=new PrintWriter(ntwk+"_x_rb_9_"+num_links);
//		10 - pw_x_rb_dt
		PrintWriter pw_x_rb_dt=new PrintWriter(ntwk+"_x_rb_dt_10_"+num_links);
//		11 - pw_x_rb_rt
		PrintWriter pw_x_rb_rt=new PrintWriter(ntwk+"_x_rb_rt_11_"+num_links);
//		12 - pw_x_rb_dt_rt
		PrintWriter pw_x_rb_dt_rt=new PrintWriter(ntwk+"_x_rb_dt_rt_12_"+num_links);

//		13 - pw_x_dt
		PrintWriter pw_x_dt=new PrintWriter(ntwk+"_x_dt_13_"+num_links);
//		14 - pw_x_dt_rt
		PrintWriter pw_x_dt_rt=new PrintWriter(ntwk+"_x_dt_rt_14_"+num_links);
//		15 - pw_x_rt
		PrintWriter pw_x_rt=new PrintWriter(ntwk+"_x_rt_15_"+num_links);
		
		
		
		for(Map.Entry<Integer, String> item: x_db.entrySet()) {
			int net=item.getKey();
			
			String x_db_=net+x_db.get(net);
			pw_x_db.println(x_db_);
			String x_db_rb_=net+x_db.get(net)+x_rb.get(net);
			pw_x_db_rb.println(x_db_rb_);
			String x_db_dt_=net+x_db.get(net)+x_dt.get(net);
			pw_x_db_dt.println(x_db_dt_);
			String x_db_rt_=net+x_db.get(net)+x_rt.get(net);
			pw_x_db_rt.println(x_db_rt_);
			
			String x_db_rb_dt_=net+x_db.get(net)+x_rb.get(net)+x_dt.get(net);
			pw_x_db_rb_dt.println(x_db_rb_dt_);
			String x_db_rb_rt_=net+x_db.get(net)+x_rb.get(net)+x_rt.get(net);
			pw_x_db_rb_rt.println(x_db_rb_rt_);
			String x_db_dt_rt_=net+x_db.get(net)+x_dt.get(net)+x_rt.get(net);
			pw_x_db_dt_rt.println(x_db_dt_rt_);
			
			String x_db_rb_dt_rt_=net+x_db.get(net)+x_rb.get(net)+x_dt.get(net)+x_rt.get(net);
			pw_x_db_rb_dt_rt.println(x_db_rb_dt_rt_);
			
			String x_rb_=net+x_rb.get(net);
			pw_x_rb.println(x_rb_);
			String x_rb_dt_=net+x_rb.get(net)+x_dt.get(net);
			pw_x_rb_dt.println(x_rb_dt_);
			String x_rb_rt_=net+x_rb.get(net)+x_rt.get(net);
			pw_x_rb_rt.println(x_rb_rt_);
			String x_rb_dt_rt_=net+x_rb.get(net)+x_dt.get(net)+x_rt.get(net);
			pw_x_rb_dt_rt.println(x_rb_dt_rt_);
			
			String x_dt_=net+x_dt.get(net);
			pw_x_dt.println(x_dt_);
			String x_dt_rt_=net+x_dt.get(net)+x_rt.get(net);
			pw_x_dt_rt.println(x_dt_rt_);
			String x_rt_=net+x_rt.get(net);
			pw_x_rt.println(x_rt_);
		}
		pw_x_db.close();
		pw_x_db_rb.close();
		pw_x_db_dt.close();
		pw_x_db_rt.close();
		pw_x_db_rb_dt.close();
		pw_x_db_rb_rt.close();
		pw_x_db_dt_rt.close();
		pw_x_db_rb_dt_rt.close();
		pw_x_rb.close();
		pw_x_rb_dt.close();
		pw_x_rb_rt.close();
		pw_x_rb_dt_rt.close();
		pw_x_dt.close();
		pw_x_dt_rt.close();
		pw_x_rt.close();
	}
	
	public static void chkLinkFeats_num_links_345(String ntwk) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(ntwk+"_"+algs[alg_idx]+"_val.txt"));
		String line="";
		HashMap<Integer, Integer> cnt=new HashMap<Integer, Integer>();
		PrintWriter pw=new PrintWriter(ntwk+"_val_rd_"+num_links+".txt");
		while ((line=br.readLine()) != null) {
			String[] lines=line.split("\t");
			String[] lines2=lines[0].split(",");			
			
			if(cnt.containsKey(lines2.length)){
				cnt.replace(lines2.length, cnt.get(lines2.length)+1);
			}
			else{
				cnt.put(lines2.length, 1);
			}
			
			if(lines2.length==3 || lines2.length==4 || lines2.length==5){
				pw.println(line);
			}
		}
		br.close();
		pw.close();
		
		boolean flag=true;
		for(Map.Entry<Integer, Integer> entry: cnt.entrySet()){
			int siz=entry.getKey();
			int val=entry.getValue();
			
			if(flag && num_links==siz)
				System.out.println(siz+" - "+val);
		}
	}
	
	public static void chkLinkFeats(String ntwk) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(ntwk+"_"+algs[alg_idx]+"_val.txt"));
		String line="";
		HashMap<Integer, Integer> cnt=new HashMap<Integer, Integer>();
		PrintWriter pw=new PrintWriter(ntwk+"_val_rd_"+num_links+".txt");
		while ((line=br.readLine()) != null) {
			String[] lines=line.split("\t");
			String[] lines2=lines[0].split(",");			
			
			if(cnt.containsKey(lines2.length)){
				cnt.replace(lines2.length, cnt.get(lines2.length)+1);
			}
			else{
				cnt.put(lines2.length, 1);
			}
			
			if(lines2.length==num_links){
				pw.println(line);
			}
		}
		br.close();
		pw.close();
		
		boolean flag=true;
		for(Map.Entry<Integer, Integer> entry: cnt.entrySet()){
			int siz=entry.getKey();
			int val=entry.getValue();
			
			if(flag && num_links==siz)
				System.out.println(siz+" - "+val);
		}
	}
	
	public static void dataGeneration(String ntwk) throws NumberFormatException, IOException, InterruptedException{
		HashMap<String, String> Node2Link=new HashMap<String, String>();
		HashMap<String, String> Link2Node=new HashMap<String, String>();
		
		(new File(ntwk+"_"+algs[alg_idx]+"_val.txt")).delete();
		
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
		for(int r=3;r<=max_links;r++){
			if(memory<nLinksets && memory<CombinatoricsUtils.binomialCoefficient(nodes.size(), r)){
				memory=(int)CombinatoricsUtils.binomialCoefficient(nodes.size(), r);
			}
			
			int num=nLinksets<memory ? nLinksets : memory;
			ArrayList<HashSet<String>> cbs=new ArrayList<HashSet<String>>();
			ArrayList<HashSet<String>> cbs_good=new ArrayList<HashSet<String>>();
			int ccnt=0;
			while(cbs_good.size()<num && cbs.size()<Integer.MAX_VALUE && cbs.size()<CombinatoricsUtils.binomialCoefficient(nodes.size(), r)){
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
					
					if(validCombi(ntwk, lks,  data)) {
						cbs_good.add(tmp);
					}
					
					System.out.println(cbs_good.size()+" - "+cbs.size()+" - "+ccnt);
				}
			}
		}
	}
	
	public static boolean validCombi(String ntwk, String str,  LinkedHashMap<String, String> data) throws NumberFormatException, IOException, InterruptedException {
		boolean res=false;
		PrintWriter pw2 = new PrintWriter(new FileOutputStream(
			    new File(ntwk+"_"+algs[alg_idx]+"_val.txt"), 
			    true /* append = true */)); 
		
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
		br=new BufferedReader(new FileReader(ntwk+"/"+ntwk+algms[alg_idx]+".txt"));
	    while ((line=br.readLine()) != null){
	    	s1.add(line);
	    }   
		br.close();
		
		
		
		System.out.println(str);
		Runtime rt = Runtime.getRuntime();
		Process pr = rt.exec("/home/durham314/Downloads/TrafficAssign-master/assign /home/durham314/Downloads/TrafficAssign-master/s_"+ntwk+"_"+algs[alg_idx]+".txt");
		pr.waitFor();
		
		HashSet<String> s2=new HashSet<String>();
		br=new BufferedReader(new FileReader(ntwk+"/"+ntwk+algms[alg_idx]+".txt"));
	    while ((line=br.readLine()) != null){
	    	s2.add(line);
	    }   
		br.close();
		
		if(!s1.equals(s2)){
			System.out.println("Good!");
			res=true;
			
			br=new BufferedReader(new FileReader(ntwk+"/"+ntwk+algms[alg_idx]+".txt"));
			
			String optimal_objective="";
		    while ((line=br.readLine()) != null){
		    	if(line.contains("<optimal objective>")){	    // <optimal objective> or <system cost>
		    		optimal_objective=(Double.parseDouble(line.split(">")[1].replaceAll("\\s+","")))+"";
//		    		pw2.println(str+"\t"+(Double.parseDouble(line.split(">")[1].replaceAll("\\s+","")))+"");
//		    		break;
		    	}
		    	
		    	if(line.contains("<system cost>")){	    // <optimal objective> or <system cost>
		    		pw2.println(str+"\t"+optimal_objective+"\t"+(Double.parseDouble(line.split(">")[1].replaceAll("\\s+","")))+"");
		    		break;
		    	}
		    }   
			br.close();
		}
		pw2.close();
		
		return res;
	}
	
	public static void bagOfWords(String ntwk) throws NumberFormatException, IOException{
		BufferedReader br = new BufferedReader(new FileReader("link_feats/"+ntwk+"_x_4F2_s_Scale"));
		String line="";
		int cnt=0;
		int feats=0;
		while ((line=br.readLine()) != null) {
			cnt++;
			feats=line.split("\t").length-1;
		}
		br.close();
		
		System.out.println(cnt+"\t"+feats);
		
		double[][] data_all=new double[cnt][feats];
		
		br = new BufferedReader(new FileReader("link_feats/"+ntwk+"_x_4F2_s_Scale"));
		while ((line=br.readLine()) != null) {
			String[] lines=line.split("\t");
			for(int i=1;i<lines.length;i++) {
				data_all[Integer.parseInt(lines[0])][i-1]=Double.parseDouble(lines[i]);
			}
		}
		br.close();
		
		double[][] data_all2=new double[cnt][feats];
		double[] feats_bag_size=new double[feats];
		for(int nfeats=0;nfeats<feats;nfeats++){			
			double[][] data=new double[cnt][1];
			for(int i=0;i<cnt;i++){
				data[i][0]=data_all[i][nfeats];
			}
			/*************************************************************/
			double[][] centroids=null;
			int[] assignments=null;
//			int[] clusterSize=null;
			// Davies–Bouldin index (the lower the better)
			double DBI=Double.MAX_VALUE;
			for(int i=2;i<=(int)Math.sqrt(cnt);i++){
//			for(int i=2;i<cnt;i++){
				KMeans xms_=new KMeans(data, i, 10000);
				double[][] centroids_=xms_.centroids();
				int[] assignments_=xms_.getClusterLabel();
				int[] clusterSize_=xms_.getClusterSize();
				double DBI_TMP=DaviesBouldinIndex(data, centroids_, assignments_, clusterSize_);
//				if(DBI>DBI_TMP && DBI_TMP>0){
				if(DBI>DBI_TMP){
					DBI=DBI_TMP;
					assignments=new int[assignments_.length];
					centroids=new double[centroids_.length][centroids_[0].length];
//					clusterSize=new int[centroids_.length];
					assignments=xms_.getClusterLabel();
					centroids=xms_.centroids();
//					clusterSize=xms_.getClusterSize();
				}
				
			}
			/*************************************************************/
			
			System.out.println(DBI+"\t"+assignments.length+"\t"+centroids.length+"\t"+centroids[0].length);		
			feats_bag_size[nfeats]=centroids.length;
			for(int i=0;i<assignments.length;i++) {
				data_all2[i][nfeats]=assignments[i];
			}
		}
		
		PrintWriter pw=new PrintWriter("link_words/"+ntwk+"_x_4F2_s_Words");
		
		pw.print(-1);
		for(int j=0;j<feats;j++){
			pw.print("\t"+feats_bag_size[j]);
		}
		pw.println();
		
		for(int i=0;i<cnt;i++){
			pw.print(i);
			for(int j=0;j<feats;j++){
				pw.print("\t"+data_all2[i][j]);
			}
			pw.println();
		}
		pw.close();
	}
	
	public static double DaviesBouldinIndex(double[][] data, double[][] centroids, int[] assignments, int[] clusterSize){
		int n=data.length; // number of data
		
		HashSet<Integer> number_of_clusters=new HashSet<Integer>();		
		for(int i:assignments){
			number_of_clusters.add(i);
		}
		
		int c=centroids.length;
		
		double[] S=new double[c];
		double[][] M=new double[c][c];
		double[][] R=new double[c][c];
		
		double[] D=new double[c];
		
		for(int i=0;i<n;i++){
			S[assignments[i]]=S[assignments[i]]+Math.squaredDistance(data[i], centroids[assignments[i]]);
		}
		
		for(int i=0;i<c && number_of_clusters.contains(i);i++){
			S[i]=Math.sqrt(S[i]/(double)clusterSize[i]);
			for(int j=0;j<i && number_of_clusters.contains(j);j++){
				M[i][j]=Math.distance(centroids[i], centroids[j]);
				M[j][i]=M[i][j];
			}
		}
		
		for(int i=0;i<c && number_of_clusters.contains(i);i++){
			for(int j=0;j<i && number_of_clusters.contains(j);j++){
				R[i][j]=(S[i]+S[j])/M[i][j];
				R[j][i]=R[i][j];
			}
		}
		
		double DB=0;
		for(int i=0;i<c && number_of_clusters.contains(i);i++){
			for(int j=0;j<c && number_of_clusters.contains(j);j++){
				if(D[i]<R[i][j] && i!=j){
					D[i]=R[i][j];
				}
			}
			DB=DB+D[i];
		}
		return DB/(double)number_of_clusters.size();
	}	
	
	public static ArrayList<HashSet<String>> rcombination(Set<String> nodes, int length, int num){
		ArrayList<HashSet<String>> cbs=new ArrayList<HashSet<String>>();
		
		while(cbs.size()<num){
			ArrayList<String> nodes_copy=new ArrayList<String>(nodes);
			HashSet<String> tmp=new HashSet<String>();
			Random r = new Random();
			
			for(int i=0;i<length;i++){
				int idx=r.nextInt(nodes_copy.size());
				tmp.add(nodes_copy.get(idx));
				nodes_copy.remove(idx);
			}
			boolean flag=true;
			for(HashSet<String> i:cbs){
				if(i.containsAll(tmp)){
					flag=false;
					break;
				}
			}
			if(flag){
				cbs.add(tmp);
			}
		}
		
		return cbs;
	}
}
