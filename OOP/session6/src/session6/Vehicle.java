package session6;

public abstract class Vehicle {
	// Note: protected: Can be accessed by subclasses or classes in the same package
	protected double topSpeed;
	protected double mass;
	
	// Constructor
	public Vehicle(double topSpeed, double mass){
		this.setTopSpeed(topSpeed);
		this.setMass(mass);
	}
	
	// Compute value
	public double calculateValue(){
		return 1000*this.topSpeed/(this.mass*0.1);
	}
	
	// Getters and setters
	public double getTopSpeed() {
		return topSpeed;
	}
	public void setTopSpeed(double topSpeed) {
		this.topSpeed = topSpeed;
	}
	public double getMass() {
		return mass;
	}
	public void setMass(double mass) {
		this.mass = mass;
	}
} 