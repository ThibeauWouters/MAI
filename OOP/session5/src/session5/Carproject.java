package session5;

import java.util.Scanner;
import session5.*;

public class Carproject {
	
	public static void main(String args []) {
		
		// create new instance of car
		Car car1 = new Car("BMW",7);

		// ask user how many miles he wants to drive
		System.out.println("Please enter how many miles you want to drive: ");
		Scanner scan = new Scanner(System.in);
		double miles = scan.nextDouble();
		scan.close();
		

		// convert the miles to kilometers
		DistanceConverter distConverter = new DistanceConverter();
		double km = distConverter.convert(miles);

		// calculate the liter fuel needed
		double litersFuel = (km/100) * car1.getFuelConsumption();
		System.out.println("Driving " + miles + " miles with this " +
				car1.getBrand() + " will consume " + litersFuel + " liters fuel.");
	}
}

