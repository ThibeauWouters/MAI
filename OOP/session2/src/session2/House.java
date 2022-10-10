package session2;

public class House {

	// Variables
	final int maxNumberOfInhabitants;
	Person[] inhabitants;
	
	
	// Constructor
	House(Person[] inhabitants, int maxNumberOfInhabitants){
		this.inhabitants = inhabitants;	
		this.maxNumberOfInhabitants = maxNumberOfInhabitants;	
	}
	
	// Methods
	public void assessLivingConditions() {
		
		if (inhabitants.length > this.maxNumberOfInhabitants) {
			System.out.println("There are too many persons living in this house.");
		} else if (inhabitants.length == this.maxNumberOfInhabitants) {
			System.out.println("The house is full.");
		} else {
			System.out.println("There is room.");
		}
	}
	
	public static void main(String[] args) {
		
		// Create persons
		Person john = new Person("John", 20, true);
		Person mary = new Person("Mary", 20, true);
		
		// Make inhabitants out of these persons
		Person[] family = new Person[] {john, mary};
		
		// Create a house and put inhabitants in the house
		House firstHouse = new House(family, 1);
		
		// Test the assessLivingConditions method
		firstHouse.assessLivingConditions();
	}
}
