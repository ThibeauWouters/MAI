package session2;

public class Recursion {

	
	// Factorial
	public static int factorial(int n) {
		
		if (n == 0) {
			return 1;
		} else {
			return n*factorial(n-1);
		}
	}
	
	// Main method
	public static void main(String[] args) {
		
		int n = 5;
		System.out.println("Testing factorial: \n " + n + " factorial equals " + factorial(n));
	}
	
}
