package team.editors;


//import java.util.HashMap;

import alignments.StandardAlignment;
import datastructures.Repository;
import team.Employee;

/**
 * This is a class implementing general aspects about employees
 * which can edit a personal copy of an alignment. Currently, only
 * the bioinformaticians are editors, but future applications can have
 * several types of editors which do not necessarily share the
 * same set of functionalities, such that there can be different subclasses
 * of this abstract editor class which allow for a diverse team.
 * 
 * @author ThibeauWouters
 */
public abstract class Editor extends Employee {

	/* Variables */
	
	/**
	 * alignment is the personal alignment that each editor can work on independently
	 */
	protected StandardAlignment alignment; 

	/* Constructor */

	/**
	 * The constructor first creates a regular Employee object. The additional argument is the
	 * initial alignment on which this editor is going to work.
	 *
	 * @param employeeType The type of employee, currently only "Bioinformatician" is valid. 
	 * @param firstName The first name of this employee. 
	 * @param lastName The last name of this employee.
	 * @param yearsOfExperience The number of years of experience of this employee.
	 * @param sa The personal alignment of this editor to work on. 
	 */
	public Editor(String employeeType, String firstName, String lastName, int yearsOfExperience, StandardAlignment sa) {
		// Create employee object
		super(employeeType, firstName, lastName, yearsOfExperience);
		// Add alignment as instance variable
		this.alignment = sa.copy();
	}
	
	/* Write data */
	/**
	 * Writes the alignment of this editor to the Reports directory of the specified Repository instance.
	 *
	 * @param repo The Repository of to which we will write the alignment, in its 
	 * Reports directory. 
	 */
	public void writeData(Repository repo) {
		repo.writeData(this);
	}

	/**
	 * Writes the score of this editor to the Reports directory of the specified Repository instance.
	 *
	 * @param repo The Repository of to which we will write the alignment, in its Reports directory.
	 */
	public void writeReport(Repository repo) {
		repo.writeReport(this);
	}

	/* Getters and setters */

	public int getScore() {
		return this.alignment.getDifferenceScore();
	}

	public void setAlignment(StandardAlignment sa) {
		this.alignment = sa.copy();
	}

	public StandardAlignment copyAlignment() {
		return this.alignment.copy();
	}
	
	public void displayAlignment() {
		this.alignment.display();
	}
}
