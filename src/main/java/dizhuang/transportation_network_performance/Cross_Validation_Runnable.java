package dizhuang.transportation_network_performance;

import java.io.IOException;

import brandeis.compressiveprivacy.libsvm.svm_train;

public class Cross_Validation_Runnable implements Runnable{
	private String param_c;
	private String param_g_ml;
	private int folder_cv;
	private Thread t;
	
	public Cross_Validation_Runnable(String param_c, String param_g_ml, int folder_cv){
		this.param_c=param_c;
		this.param_g_ml=param_g_ml;
		this.folder_cv=folder_cv;
	}

//	@Override
	public void run() {
		// TODO Auto-generated method stub
		try {	
			String[] trainArgs = {"-v", folder_cv+"","-g", param_g_ml+"", "-c", param_c+"", //"-h", "0", 
				"tr_data", "tr_model"};
		
			svm_train.main(trainArgs, param_c+"\t"+param_g_ml);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void start()
	{
		if(t==null)
		{
			t=new Thread(this);
			t.start();
		}
	}

}
