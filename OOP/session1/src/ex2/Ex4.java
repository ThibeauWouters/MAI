package ex2;


public class Ex4 {
	/**
	 * This is actually exercise 5, but it also shows exercise 4. 
	 */
	// Define global variables
	static double a = 1.8;
	static int b = 32;
	
	
	public static void main(String[] args) {
		double fahrenheit = changeToFahrenheit(37.0);
		double celsius = changeToCelsius(37);
		
		System.out.println("Celsius: " + celsius + "\nFahrenheit:" + fahrenheit);
	}

	public static int changeToFahrenheit(double celsius) {
		// An application that converts Celsius to Fahrenheit.
		int fahrenheit = (int) (celsius * a) + b;
		return fahrenheit;
	}
	
	public static double changeToCelsius(double fahrenheit) {
		// An application that converts Fahrenheit to Celsius.
		double celsius = (fahrenheit - b)/a;
		return celsius;
	}
}
