package session3;

public class Employee {
	String employeeType;
	int nbYearsWorked;
	double baseWage;
	double wage;

	public Employee(String employeeType, int nbYearsWorked) {
		this.employeeType = employeeType.toUpperCase();
		this.nbYearsWorked = nbYearsWorked;
	}
	
	double getBaseWage() {
		// Gets base wage of this employee based on the type
		switch (this.employeeType) {
		case "CLERK" : baseWage = 1000; break;
		case "MIDLEVELMANAGER" : baseWage = 2000; break;
		case "CIO" : 
		case "CFO" : baseWage = 3000; break;
		case "CEO" : baseWage = 5000; break;
		default: baseWage = 0; break;
		}
		
		return baseWage;
		
	}
	
	public double calculateWage() {
		return wage = this.getBaseWage()*(1 + nbYearsWorked*0.1);
	}
	
	public static void main(String[] args) {
		Employee bigBoss = new Employee("ceo", 20);
		Employee peasant = new Employee("ict dude", 100);
		
		System.out.println("The first is " + bigBoss.employeeType + " and wage is " + bigBoss.calculateWage());
		System.out.println("The first is " + peasant.employeeType + " and wage is " + peasant.calculateWage());
		
	}
}
