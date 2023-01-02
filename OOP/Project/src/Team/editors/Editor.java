package team.editors;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

//import java.util.HashMap;

import alignments.StandardAlignment;
import datastructures.Repository;
import team.Employee;

/**
 * This is a class implementing general aspects about employees
 * which can edit a personal copy of an alignment. Currently, only
 * the bioinformaticians are editors, but future applications can have
 * several different types of editors which do not necessarily share the
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
		this.setHasRepositoryAccess(false);
	}

	/* Methods */

	/**
	 * Method for an editor to write down a copy of his/her personal alignment to 
	 * a specified folder containing all reports.
	 * 
	 * @param reportsDirectory A File object pointing towards a directory storing all reports.
	 */
	public void writeData(File reportsDirectory) {
		// Get the path and create a relative path towards the desired file
		String currentPath = reportsDirectory.getAbsolutePath();
		File path = new File(currentPath, this.getFileString(Repository.getDataExtension()));
		String fileName = path.getAbsolutePath();
		System.out.println(this.getName() + " saves his/her alignment to " + fileName);

		// Call auxiliary method to write down alignment to this file
		Repository.copyAlignmentToFile(fileName, this.copyAlignment());
	}

	/**
	 * Method for an editor to write down the difference score of his/her 
	 * personal alignment to a specified folder containing all reports.
	 * 
	 * @param reportsDirectory A File object pointing towards a directory storing all reports.
	 */
	public void writeReport(File reportsDirectory) {
		// Get the current path and construct relative path pointing to desired file
		String currentPath = reportsDirectory.getAbsolutePath();
		File path = new File(currentPath, this.getFileString(Repository.getReportExtension()));
		String fileName = path.getAbsolutePath();
		System.out.println(this.getName() + " saves his/her score to " + fileName);

		// Write down the score to the desired file
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			bw.write(this.getName() + "\n");
			bw.write(this.getScore() + "\n");
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found when writing score.");
			fnf.printStackTrace();
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing score.");
			ioe.printStackTrace();
		} 
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
