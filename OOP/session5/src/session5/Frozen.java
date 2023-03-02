package session5;

public class Frozen {
	public static void main (String [] args){
		//int [] clientid = {1,2,3,4,5};
		String [] clientname = {"Anna", "Elsa", "Olaf", "Hans", "Kristoff"};
		double [] accountbalance = {1000, 5000, 9, 55, 23};
		for(int i = 1; i<=5; i++)
		{
			try {
				System.out.println(clientname[i] + " has " +accountbalance[i] + " dollars. \n");
			} catch(ArrayIndexOutOfBoundsException exc) {
				System.out.println("Hey we caught the exception, namely: " + exc);
				System.out.println("Stopping the program.");
				System.exit(0);
				}
		}
	}
}