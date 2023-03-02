package session6;

public abstract class Employee extends NormalPerson{
	
	private final int employeeID;
	
	public Employee(String name, String gender, int employeeID) {
		super(name, gender);
		this.employeeID = employeeID;
	}
	
	public abstract double calculateSalary();
	
	// Getters and setters
	
	public int getEmployeeID() {
		return this.employeeID;
	}
	
//	public void setEmployeeID(int newID) {
//		this.employeeID = newID;
//	}
}
