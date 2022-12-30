package team.editors;

//import java.util.HashMap;

import alignments.StandardAlignment;
import datastructures.Repository;
import team.Employee;

public abstract class Editor extends Employee {
	
	/* Variables */
	protected StandardAlignment alignment; 
	
	/* Constructor */
	
	public Editor(String employeeType, String firstName, String lastName, int yearsOfExperience, StandardAlignment sa) {
		
		// Create employee object
		super(employeeType, firstName, lastName, yearsOfExperience);
		// Add alignment as instance variable
		this.alignment = sa.copy();
	}
	
	/* Methods */
	
	/* Shared editor methods */
	
	public StandardAlignment copyAlignment() {
		
		// Return deepcopy of alignment
		return this.alignment.copy();
	}
	
	/* Methods to write data to files */
	
	public void writeData(Repository repo) {
		repo.writeData(this);
	}
	
	public void writeReport(Repository repo) {
		repo.writeReport(this);
	}
	
	/* Getters and setters */
	
	public int getScore() {
		// Return the score of this editor's current alignment
		return this.alignment.getDifferenceScore();
	}
	
	public void setAlignment(StandardAlignment sa) {
		this.alignment = sa.copy();
	}
}
