package session5;

import java.util.Scanner;
import java.util.InputMismatchException;

public class BMIcalculator{
	// Main method
	public static void main(String[] args) throws InputMismatchException{
		// Call the BMI calculator
		try {
			double BMI = calculateBMI();
			// Print BMI to screen
			System.out.println("Your BMI is " + BMI + ".");
		} catch (ArithmeticException ae) {
			System.out.println("We caught an arithmetic exception. Stopping the program.");
			System.exit(0);
		} catch (InputMismatchException ae) {
			System.out.println("We caught an input mismatch exception. Stopping the program.");
			System.exit(0);
		}
	}


	// method calculating BMI
	public static double calculateBMI() throws InputMismatchException, ArithmeticException{
		double BMI, weight, height;
		
		// Get weight and height from the user
		try(Scanner scan = new Scanner(System.in)){
			System.out.print("Please give your weight: ");
			weight = scan.nextDouble();
			System.out.print("Please give your height: ");
			height = scan.nextDouble();
			// Check if input was not negative, or too large
			if(weight <= 0 || height <= 0) {
				throw new ArithmeticException();
			} else if(weight >= 200 || height >= 200) {
				throw new ArithmeticException();
			} else {
				// Otherwise, compute BMI
				BMI = weight/(height*height);
				return BMI;
			}
		}
	}
}
