package Team;

public class Employee {
	
	/* Variables */
	protected String employeeType;
	protected String firstName;
	protected String lastName;
	protected int yearsOfExperience;
	
	/* Constructor */
	public Employee(String employeeType, String firstName, String lastName, int yearsOfExperience) {
		this.employeeType      = employeeType;
		this.firstName         = firstName;
		this.lastName          = lastName;
		this.yearsOfExperience = yearsOfExperience;
	}
	
	/* Getters and setters */
	
	public String getEmployeeType() {
		return this.employeeType;
	}
	
	public String getFirstName() {
		return this.firstName;
	}
	
	public String getLastName() {
		return this.lastName;
	}
	
	public String getName() {
		return this.firstName + " " + this.lastName;
	}
	
	public int getYearsOfExperience() {
		return this.yearsOfExperience;
	}
	
	public void incrementYearsOfExperience() {
		// In case it's an employee's work anniversary, increase its years of experience
		this.yearsOfExperience += 1;
	}
	

}
