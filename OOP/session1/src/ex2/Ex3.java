package ex2;



public class Ex3 {

	public static void main(String[] args) {

		double[] firstArray = new double[] {1, 2, 3};
		double[] secondArray = new double[] {3, 2, 1};
		
		
		// First method
		boolean[] truthArray = new boolean[firstArray.length];
		
		for(int i = 0; i < firstArray.length; i++) {
			
			truthArray[i] = firstArray[i] == secondArray[i];
			
		}
		}
		

}
