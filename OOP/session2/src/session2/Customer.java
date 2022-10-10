package session2;

public class Customer {

	// Variables
	int age;
	String name;
	private boolean isMarried; // instance variable
	static final int adultAge = 18; // class variable
	
	// Constructor
	Customer(String n, int a, boolean maritalStatus)
	{
		age = a;
		name = n;
		isMarried = maritalStatus;	
	}
	
	// Methods
	
	// Method to check if customer is an adult or not
	boolean isAdult() {
		return (age >= adultAge);
	}
	
	// Main method
	public static void main(String[] args) {
		
		Customer myFirstCustomer = new Customer("Maria", 19, false); 
		
	}
	
	
}
