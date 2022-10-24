package session5;

public class DistanceConverter {
	
	// constructor method
	public DistanceConverter() {
	}
	
	// method to calculate the BMI
	public double convert(double dist_miles) {
		// define conversion factor
		double oneMileInKM = 1.609;
		
		// conversion
		final double dist_km;
		dist_km = dist_miles * oneMileInKM;
		
		return dist_km;
	}
}
