package team;

import java.util.ArrayList;

//import alignments.*;
import team.editors.Bioinformatician;
import team.editors.Editor;

/**
 * This file reads in the team.txt file and creates the employees. 
 * It can be used in future extensions of the application to implement
 * general functionalities refering to the bioinformatics team as a whole 
 * as well. 
 * @author ThibeauWouters
 *
 */
public class Team {

	/**
	 * team is an ArrayList holding all Employee objects belonging to this team.
	 * An ArrayList is flexible in size, meaning future applications can easily
	 * perform new operations such as adding or removing team members.
	 */
	ArrayList<Employee> team = new ArrayList<Employee>();	
	
	/* Constructor */
	
	/**
	 * The constructor directly takes an existing ArrayList of employees and stores it as instance variable.
	 * 
	 * @param fileName Name of the .txt file holding the details of the team to be created.
	 * @param sa A StandardAlignment object which is going to be the initial alignment for editors.
	 */
	public Team(ArrayList<Employee> teamArrayList) {
		this.team = teamArrayList;
	}

	public void removeEmployee(Employee employee) {
		// Remove an employee from the team
		this.team.remove(employee);
	}
	
	public void addEmployee(Employee employee) {
		// Add an employee to the team
		this.team.add(employee);
	}
	
	
	/* Getters and setters */
	
	public ArrayList<Employee> getTeam(){
		return this.team;
	}
	
	/**
	 * Auxiliary method showing the current team members in an organized manner.
	 * Used just for showing the operation of the application in the main method.
	 */
	public void displayTeamMembers() {
		for (Employee empl : this.getTeam()) {
			System.out.println("Employee name:    " + empl.getName());
			System.out.println("Employee type:    " + empl.getEmployeeType());
			System.out.println("Experience (yrs): " + empl.getYearsOfExperience());
		}
	}
	
	/* Getters to return lists for specific employee types */
	
	// Note: the methods below are mainly for illustration purposes in the main method,
	// and may not be useful outside of their application in the main method!!!
	
	public ArrayList<Editor> getEditors(){
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
