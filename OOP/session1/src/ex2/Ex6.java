package ex2;

import java.util.Arrays;

public class Ex6 {
	
	/**
	 * This is actually exercise 7, but it also shows exercise 6.
	 * @param array
	 * @return
	 */
	
	public static float meanOfArray(float[] array) {
		
		// Calculate the sum of all elements
		
		float sum = 0;
		
		for(int i = 0; i < array.length; i++) {
			sum += array[i];
		}
		
		float median = sum/array.length;
		
		// If desired as an array: (see exercise 6)
		//int medianAsInteger = (int) median;
		
		return median;
	}
	
	
	public static void main(String[] args) {
		
		// Make up an array:
		// Note: if we want floats, put an f at the back of the number!
		float[] numbers = new float[] {1.1f, 2.2f, 3.3f, 6.14f, 7.5f};

		float medianAsInteger = meanOfArray(numbers);
		System.out.println("The given numbers are " + Arrays.toString(numbers));
		System.out.println("The median of the numbers as integer is " + medianAsInteger);
		
	}

}
