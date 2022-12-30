package team;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.Scanner;

import alignments.*;
import team.editors.Bioinformatician;
import team.editors.Editor;

public class Team {
	
	private String fileName;
	ArrayList<Employee> team = new ArrayList<Employee>();	
	
	public Team(String fileName, StandardAlignment sa) {
		
		// Save the filename for future convenience
		this.fileName = fileName;
		
		// Check if fileName corresponds to a TXT file (final 4 characters are ".txt")
		// TODO can we throw some exception for this?
		String extension = fileName.substring(fileName.length()-4, fileName.length());
		if (!(extension.equals(".txt"))) {
			System.out.println("Constructor takes .txt files only. Exiting constructor.");
			System.exit(0);
		}
		
		// TODO: is this ok in terms of getting the filepath etc?
//		System.out.println("Teams file entered: " + fileName);
		
		// Scan all lines
		try(Scanner input = new Scanner(new FileReader(fileName));){ 
			while(input.hasNext()) { 
				try {
					// Read in the next lines and define a new employee with it
					Employee newEmployee;
					
					// Read the contents of this employee
					String employeeType               = input.next();
					String firstName                  = input.next();
					String lastName                   = input.next();
					String yearsOfExperienceString    = input.next();
					
					// Try to parse the final field as an integer, otherwise use "0" as default number of years of experience
					int yearsOfExperience = 0;
					try {
						yearsOfExperience = Integer.parseInt(yearsOfExperienceString);
						} catch (NumberFormatException e) {
							System.out.println("Could not interpret yearsOfExperience as a number. Defaulting to 0.");
						}
					
					// Call the appropriate constructor, depending on the employee type:
					switch (employeeType) {
					case "TeamLead":         newEmployee = new TeamLead(firstName, lastName, yearsOfExperience); break;
					case "Bioinformatician": newEmployee = new Bioinformatician(firstName, lastName, yearsOfExperience, sa); break;
					case "TechnicalSupport": newEmployee = new TechnicalSupport(firstName, lastName, yearsOfExperience); break;
					// In future updates, one can add additional employee types here . . . 
					default:                 newEmployee = new Employee("Unknown", firstName, lastName, yearsOfExperience); break;
					}
					
					// Add this new employee to the team
					this.addEmployee(newEmployee);
					
				} catch (InputMismatchException mismatch) {
					// In case the input is invalid:
					System.out.println("Invalid entry in file found, stop reading input."); 
					break; 
				} 
			}  
		} catch (FileNotFoundException e) {
			// In case the rights to file are invalid:
			// TODO: is this indeed the error in case we don't have the right to access the file?
			System.out.println("Error: file not found."); 
			System.out.println("Exiting program");
			System.exit(0); 
		} catch (Exception e) {
			// Catch any other exception
			System.out.println("Unexpected error occurred: " + e);
			System.out.println("Exiting program");
			System.exit(0); 
		}
		
	}
	
	/* Getters and setters */

	public String getFileName() {
		return this.fileName;
	}
	
	public void removeEmployee(String name) {
		/*
		 * Remove an employee from the team, given the name. Assume every name (first + last) is unique.
		 */
		
		// Iterate over current employees in the team:
		for (Employee empl : this.team) {
			// If their name matches the one given, remove from the team
			if (empl.getName().equals(name)){
				System.out.println(name + " ... You're fired!!!");
				this.team.remove(empl);
			}
		}
	}
	
	public void removeEmployee(Employee employee) {
		/*
		 * Remove an employee from the team, given the object.
		 */
		
		// Iterate over current employees in the team:
		String name = employee.getName();
		this.removeEmployee(name);
	}
	
	public void addEmployee(Employee empl) {
		// Add an Employee Object to the team
		this.team.add(empl);
	}
	
	public ArrayList<Employee> getTeam(){
		return this.team;
	}
	
	public void displayTeamMembers() {
		// Iterate over current employees in the team:
		for (Employee empl : this.getTeam()) {
			System.out.println("Employee name:    " + empl.getName());
			System.out.println("Employee type:    " + empl.getEmployeeType());
			System.out.println("Experience (yrs): " + empl.getYearsOfExperience());
		}
	}
	
	/* Getters to return specific employee types */
	
	// Note: the methods below are mainly for illustration purposes in the main method
	
	public ArrayList<Editor> getEditors(){
		// Creates an ArrayList of all the editors of the current team
		
		// Create empty list
		ArrayList<Editor> result = new ArrayList<Editor>();
		
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof Editor) {
				Editor editor = (Editor) employee;
				result.add(editor);
			}
		}
		return result;
	}
	
	public ArrayList<Bioinformatician> getBioinformaticians(){
		// Creates an ArrayList of all the editors of the current team
		
		// Create empty list
		ArrayList<Bioinformatician> result = new ArrayList<Bioinformatician>();
		
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof Bioinformatician) {
				Bioinformatician editor = (Bioinformatician) employee;
				result.add(editor);
			}
		}
		return result;
	}
	
	public ArrayList<TeamLead> getTeamLeads(){
		// Creates an ArrayList of all the editors of the current team
		
		// Create empty list
		ArrayList<TeamLead> result = new ArrayList<TeamLead>();
		
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof TeamLead) {
				TeamLead editor = (TeamLead) employee;
				result.add(editor);
			}
		}
		return result;
	}
	
	public ArrayList<TechnicalSupport> getTechnicalSupports(){
		// Creates an ArrayList of all the editors of the current team
		
		// Create empty list
		ArrayList<TechnicalSupport> result = new ArrayList<TechnicalSupport>();
		
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof TechnicalSupport) {
				TechnicalSupport editor = (TechnicalSupport) employee;
				result.add(editor);
			}
		}
		return result;
	}

}
