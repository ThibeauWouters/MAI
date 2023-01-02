package team;

/**
 * This abstract class implements the generic aspects of any employee, such as their name,
 * and future extensions can add more applications here. 
 * @author thibe
 *
 */
public abstract class Employee {
	
	/* Variables */
	/**
	 * employeeType is a string denoting the job of this employee (currently: Bioinformatician,
	 * TeamLead or TechnicalSupport only).
	 */
	private final String employeeType;
	/**
	 * firstName stores the first name of this employee
	 */
	private final String firstName;
	/**
	 * lastName stores the last name of this employee
	 */
	private final String lastName;
	/**
	 * yearsOfExperience is an integer denoting the number of years of
	 * experience of this employee
	 */
	protected int yearsOfExperience;
	/**
	 * hasRepositoryAccess is a boolean denoting whether or not this employee can call 
	 * functions that are implemented in the Repository and use the stored copies there.
	 * Note that this field will be initialized by the constructors of subclasses.
	 */
	private boolean hasRepositoryAccess;
	
	/* Constructor */
	
	/**
	 * Simple constructor which saves all instance variables. See above for more information.
	 */
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
	
	public int getYearsOfExperience() {
		return this.yearsOfExperience;
	}
	
	public void setYearsOfExperience(int years) {
		this.yearsOfExperience = years;
	}
	
	public void incrementYearsOfExperience() {
		System.out.println("Work anniversary! Congratulations, " + this.getName());
		this.yearsOfExperience += 1;
	}
	
	public boolean hasRepositoryAccess() {
		return hasRepositoryAccess;
	}

	public void setHasRepositoryAccess(boolean hasRepositoryAccess) {
		this.hasRepositoryAccess = hasRepositoryAccess;
	}
	
	/**
	 * Additional auxiliary method which simply returns the full name of a person.
	 * 
	 * @return The full name of the person (with a space added between first
	 * and last name for visualization purposes)
	 */
	public String getName() {
		return this.getFirstName() + " " + this.getLastName();
	}
	
	/**
	 * Additional auxiliary method which creates a string using this employee's name
	 * which is used frequently for all kinds of applications where employees have to
	 * save contents to a file of which the filename uses their name. An additional String
	 * can be given as argument to further inform about the contents of that file.
	 * 
	 * @param inputString Appends an additional string at the end, detailing the contents of the file.
	 * @return A String which can be used used as filename.
	 */
	public String getFileString(String inputString) {
		return this.getFirstName() + "_" + this.getLastName() + inputString;
	}
	
	/**
	 * Additional auxiliary method which creates a string using this employee's name
	 * which is used frequently for all kinds of applications where employees have to
	 * save contents to a file of which the filename uses their name. 
	 * 
	 * @return A String which can be used used as filename.
	 */
	public String getFileString() {
		return this.getFileString("");
	}
}
