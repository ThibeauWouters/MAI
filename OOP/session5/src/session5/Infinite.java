package session5;
import java.util.ArrayList;
import java.lang.OutOfMemoryError;


public class Infinite {

	//	public static void main (String[] args) {
	//		ArrayList<String> myStrings = new ArrayList<String>();
	//		try {
	//			for(int i=0;i>=0;i++) {
	//				myStrings.add("String number: " + i);
	//				myStrings.add("String number: " + i);
	//				myStrings.add("String number: " + i);
	//			}	
	//		} catch (OutOfMemoryError oome) {
	//			System.out.println("Headache");
	//			System.exit(0);
	//		}
	//	}	


	// SOLUTION
	public static void main(String[] args) {
		ArrayList<String> myStrings=new ArrayList<String>();
		int i=1;
		try{
			while(i>0)
			{
				myStrings.add("String number:"+ i);
				myStrings.add("String number:"+ i);
			}
		}
		catch(OutOfMemoryError e)
		{
			System.out.println("Memory is full!");
		}
	}
}
