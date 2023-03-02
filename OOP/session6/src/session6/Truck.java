package session6;

public class Truck extends Vehicle{
	private double maxPayload;
	
	public Truck(double topSpeed, double mass, double maxPayload) {
		// Call vehicle to set topSpeed and mass
		super(topSpeed, mass);
		// Now set the max payload
		this.maxPayload = maxPayload;
	}
	
	
	// The value of a truck is calculated the same way as the value of vehicle, but multiplied by (maxPayload/1000).
	@Override
	public double calculateValue() {
		double vehicleValue = super.calculateValue();
		return vehicleValue*(this.maxPayload/1000);
	}
	
	// Getters and setters
	public double getMaxPayload(){
		return this.maxPayload;
	}
	
	public void setMaxPayload(double newMaxPayload){
		this.maxPayload = newMaxPayload;
	}
}