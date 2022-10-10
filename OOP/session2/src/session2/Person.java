package session2;

public class Person {

	// Variables
	String name;
	int age;
	boolean isMarried;
	Person spouse; // defaulted to "null"
	
	
	// Constructor
	Person(String personName, int personAge, boolean personMaritalStatus){
		name = personName;
		age = personAge;
		isMarried = personMaritalStatus;
	}
	
	/*
	 *  Methods
	 */
	
	// Marry two persons, and add spouse for this
	public static void marry(Person person1, Person person2) {
		person1.isMarried = true;
		person1.spouse = person2;
		
		person2.isMarried = true;
		person2.spouse = person1;
	}
	
	// Clone a person
	public static Person clonePerson(Person p) {
		
		// Look at the solutions - different way, without any arguments!!!
		
		Person clone = new Person(p.name, p.age, p.isMarried);
		return clone;
		
	}
	
	// Main method
	public static void main(String[] args) {
		
		// Create persons
		Person john = new Person("John", 20, false);
		Person mary = new Person("Mary", 20, false);
		
		// Check the clone method
		Person evilJohn = clonePerson(john);
		System.out.println(evilJohn.age);
		
		// Marry them
		System.out.println(john.isMarried);
		System.out.println(john.spouse);
		marry(john, mary);
		System.out.println(john.isMarried);
		System.out.println(john.spouse);
	}
	
}
