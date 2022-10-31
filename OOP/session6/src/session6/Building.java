package session6;

public class Building {

	private int availableSpace;
	private String address;
	
	public Building(int availableSpace, String address){
		// Better to have a simpler constructor and do this complex logic elsewhere
		if (availableSpace < 0) {
			System.out.println("The availablespace given was negative. Defaulting to 0.");
			availableSpace = 0;
		}
		this.availableSpace = availableSpace;
		this.address = address;
	}
	
	public Building(int availableSpace) {
		this(availableSpace, "unknown"); // use "this" to call other constructor with different arity!
	}
	
	// toString() method
	@Override
	public String toString() {
		return "This is a building with " + this.availableSpace + " square meters of office space located at " + this.address;
	}
	
	// Getters
	
	public int getAvailableSpace() {
		return this.availableSpace;
	}
	
	public String getAddress() {
		return this.address;
	}
	
	// Setters
	
	//public void setAvailableSpace(int newAvailableSpace) {
	//	this.availableSpace = newAvailableSpace;
	//}
	
	// Better version:
	public void setAvailableSpace(int newAvailableSpace){
		if(newAvailableSpace > 0){
			this.availableSpace = newAvailableSpace;
		} else {
			throw new IllegalArgumentException("Available space must be larger than zero.");
		}
	}

	public void setAddress(String newAddress) {
		this.address = newAddress;
	}

	public static void main(String[] args) {
		Building b = new Building(40, "College Premonstreit te Leuven");
		String stringy = b.toString();
		System.out.println(stringy);
	}
}
