package session5;


public class Car {
	String brand;
	double fuelConsumption;
	
	// constructor method
	public Car(String br, double fc){
		brand = br;
		fuelConsumption = fc;
	}
	
	// methods to return brand and fuelConsumption
	public String getBrand(){
		return brand;
	}
	public double getFuelConsumption(){
		return fuelConsumption;
	}
}