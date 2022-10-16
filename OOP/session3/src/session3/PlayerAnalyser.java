package session3;

import java.util.Arrays;
public class PlayerAnalyser {
	public static int minGoalsScored(int[] nbGoalsScored){
		// Calculate the min
		
		int lowest = nbGoalsScored[0];
		for (int number : nbGoalsScored) {
			if (number < lowest) {
				lowest = number;
			}
		}
		
		return lowest;
	}
	public static int maxGoalsScored(int[] nbGoalsScored){
		// Calculate the max
		
		int highest = nbGoalsScored[0];
		for (int number : nbGoalsScored) {
			if (number > highest) {
				highest = number;
			}
		}
		
		return highest;
	}
	public static double meanGoalsScored(int[] nbGoalsScored){
		// Calculate the mean
		
		double sum = 0;
		double mean;
		
		for (int number : nbGoalsScored) {
			sum += number;
		}
		
		mean = sum/((double) nbGoalsScored.length);
		
		return mean;
	}
	public static double medianGoalsScored(int[] nbGoalsScored){
		//Arrays.sort sorts a given list in ascending order
		Arrays.sort(nbGoalsScored);
		double median;
		
		int middleIndex = nbGoalsScored.length/2;
		//System.out.println(middleIndex);
		
		// If even number of elements:
		if (nbGoalsScored.length % 2 == 0) {
			median = (nbGoalsScored[middleIndex] + nbGoalsScored[middleIndex - 1])/2.0;
		} else {
			median = nbGoalsScored[middleIndex];
		}
		
		return median;
	}
	public static void main(String[] args){
		
		// Goals:
		int[] testSet1 = {0, 10, 2, 3, 0, 1};
		int[] testSet2 = {1, 3, 2, 8, 0};
		
		// Tests:
		System.out.println("First set");
		System.out.println(minGoalsScored(testSet1));
		System.out.println(maxGoalsScored(testSet1));
		System.out.println(meanGoalsScored(testSet1));
		System.out.println(medianGoalsScored(testSet1));
		
		System.out.println("Second set");
		System.out.println(minGoalsScored(testSet2));
		System.out.println(maxGoalsScored(testSet2));
		System.out.println(meanGoalsScored(testSet2));
		System.out.println(medianGoalsScored(testSet2));
	}
}
