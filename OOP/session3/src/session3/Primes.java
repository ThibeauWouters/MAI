package session3;

import java.util.*;  

public class Primes {
	
	public static void main(String[] args) {
		
		int number = 20;
		
		ArrayList<Integer> primesFor = getFirstPrimesUsingForLoops(number);
		ArrayList<Integer> primesWhile = getFirstPrimesUsingWhileLoops(number);
		
		boolean areEqual = primesFor.equals(primesWhile);
		
		System.out.println(primesFor);
		System.out.println(primesWhile);
		System.out.println(areEqual);
		
	}

	public static ArrayList<Integer> getFirstPrimesUsingForLoops(int nbPrimes){
		/**
		 * Function as given in the exercise session, using for loops
		 */
		
		boolean isPrime;
		
		// Make an empty array list, then add 2 to it
		ArrayList<Integer> primeNumbers = new ArrayList<Integer>();
		primeNumbers.add(2);
		
		// Search for primes, using FOR LOOPS
		for(int i = 3; primeNumbers.size()<nbPrimes; i++){
			isPrime = true;
			for(int j=i-1; j >= 2; j--){
				if(i%j == 0){
					isPrime = false;
				}
			}
			if(isPrime){
				primeNumbers.add(i);
			}
		}
		return primeNumbers;
	}
	
	
	public static ArrayList<Integer> getFirstPrimesUsingWhileLoops(int nbPrimes){
		/**
		 * Function which the exercise wants you to program
		 */
		
		boolean isPrime;
		
		// Make an empty array list, then add 2 to it
		ArrayList<Integer> primeNumbers = new ArrayList<Integer>();
		primeNumbers.add(2);
		
		// Search for primes, using WHILE LOOPS
		int i = 3;
		
		while (primeNumbers.size() < nbPrimes) {
			isPrime = true;
			int j = i - 1;
			
			// Check numbers below i to find divisors
			while (j >= 2) {
				if(i%j == 0) {
					isPrime = false;
					break;
				}
				j--;
			}
			
			if (isPrime) {
				// System.out.println("Found a prime, namely " + i);
				primeNumbers.add(i);
			}
			
			// Go to the next number
			i++;
		}

		return primeNumbers;
	}

}
