package ex2;

import java.util.Arrays;

/**
 * 
 * @author thibe
 *
 */

public class Ex2 {

	public static void main(String[] args) {
		
		// 1st method: 
		String[] nameArrayTest;
		nameArrayTest = new String[5];
		
		nameArrayTest[0] = "John";
		nameArrayTest[1] = "Anna";
		nameArrayTest[2] = "Bob";
		nameArrayTest[3] = "Peter";
		nameArrayTest[4] = "Helen";
		
		
		// 2nd method - faster!
		String[] newArray = new String[] {"John", "Anna", "Bob", "Peter", "Helen"};
		
		// To print whole arrays, use this (source: https://stackoverflow.com/questions/409784/whats-the-simplest-way-to-print-a-java-array)
		
		// Q: can't I print a whole array?
		System.out.println(nameArrayTest);
		
		System.out.println(Arrays.toString(newArray));
		
		// What about primitives? Same problem over there?
		
		int[] integers = new int[] {1,2,3,4,5};
		
		System.out.println(integers);
		System.out.println(Arrays.toString(integers));
		
		
		// EXERCISE 2
		
		// Create the arrays
		String[] countArray = new String[] {"first", "second", "third"};
		String[] nameArray = new String[] {"John", "Anna", "Bob"};
		
		// Loop and print:
		
		for(int i = 0; i < countArray.length; i++) {
			
			System.out.println("The " + countArray[i] + " name is " + nameArray[i]);
		}
	}

}
