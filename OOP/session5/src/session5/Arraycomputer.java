package session5;

public class Arraycomputer {

	public static void main(String[] args) {
		int[] myArray = {2, 6, 8, 1, 9, 0, 10, 23, 7, 5, 3};
		int ratio;
		
		for (int i=0;i<myArray.length;i++) {
			try {
				ratio = 10/myArray[i];
				if(ratio % 2 == 0) {
					System.out.println("The number is even and equal to " + ratio);
				} else {
					System.out.println("The number is odd and equal to " + ratio);
				}
			} catch(ArithmeticException ae){
				System.out.println("There was a division by zero.");
			}
		}

	}
}
