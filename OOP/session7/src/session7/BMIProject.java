package session7;

import java.util.Scanner;
public class BMIProject {
	public static void main(String [] args){
		// declare variables
		String name = new String();
		double weight = 60;
		double height = 1.70;
		double BMI;
		// Ask the user for inputs
		Scanner scan = new Scanner(System.in);
		System.out.println("Please enter name");
		name = scan.nextLine();
		System.out.println("Please enter weight [kg]");
		weight = scan.nextDouble();
		System.out.println("Please enter height [m]");
		height = scan.nextDouble();
		// calculate BMI
		BMI = calculateBMI(weight,height);
		// Here: Commands formatting the output and print to screen
		System.out.format("%s weighs %1.3f kg and measures %1.2f m. %n The BMI is %f BMI", name, weight, height, BMI);
		scan.close();
	}
	// method calculating BMI
	public static double calculateBMI(double w,double h){
		double BMI;
		BMI = w/(h*h);
		return BMI;
	}
}