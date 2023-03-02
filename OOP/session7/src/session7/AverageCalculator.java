package session7;

import java.util.Scanner;
//import java.util.ArrayList;
import java.util.InputMismatchException;

/**
 * Write a program that asks the user to enter numbers until he enters 0 
 * and then calculates the average of the numbers. Implement it to catch possible exceptions.
 * @author thibe
 */

public class AverageCalculator {

	public static void main(String[] args) {
		double sum = 0, count = 0, mean = 0;
		double number;
		// Ask the user to input stuff
		System.out.println("Please enter a number or 0 to stop");
		try(Scanner scan = new Scanner(System.in);){
			number = scan.nextDouble();
			count++;
			sum += number;
			while(number != 0) {
				number = scan.nextDouble();
				count++;
				sum += number;
			}
			}
		catch (InputMismatchException ime) {
			System.out.println("Hey!!!! That's not a number!!! I QUIT!");
			//scan.close();//not needed if resources are in the try block???
		}finally {
			if(count == 0) {
				System.out.println("We did not receive any numbers.");
			} else {
				mean = sum/count;
				System.out.format("Hey your mean is %f. %n", mean);
			}
		}
	}
	// Not necessary
//	
//	public static float calculateMean(ArrayList<Float> numbers) {
//		float sum = 0;
//		for(int i = 0; i < numbers.size(); i++) {
//			sum += numbers.get(i);
//		}
//		
//		float mean = sum/numbers.size();
//		return mean;
//	}
}
